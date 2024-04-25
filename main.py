import os
from copy import deepcopy
from dataclasses import field, dataclass
import dataclasses
from datetime import datetime
import json
from pathlib import Path
from time import sleep, time
import traceback
import uuid
import streamlit as st
import openai
from openai.types.chat import ChatCompletionMessageParam
import yaml
import altair as alt
import pandas as pd

MODELS = [
    None,
    "gpt-4-0125-preview",
    "gpt-3.5-turbo-0125",
]
TEACHER_NAME = "Camille"

BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)
BACKUP_FREQUENCY = 5 * 60  # seconds
TIME_PER_QUESTION = 3 * 60


@dataclass(frozen=True)
class Example:
    original: str
    student: str
    feedback: str


@dataclass(frozen=True)
class Exercise:
    name: str
    instructions: str
    """Initial instructions shown at the start of the exercise"""
    variations: list[str]
    """Variations using the same instructions"""
    system_prompt: str
    """System prompt given to gpt for feedback"""
    examples: list[Example]
    """Examples of triplets (question, answer, feedback), to guide gpt-4"""
    disable_timer: bool = False

    @classmethod
    def from_yaml(cls, data: dict):
        data["examples"] = [Example(**ex) for ex in data["examples"]]
        return cls(**data)


EXERCISES = [Exercise.from_yaml(d) for d in yaml.safe_load_all(Path("exercises.yaml").read_text())]
# Filter for HIDDEN exercises
EXERCISES = [exo for exo in EXERCISES if not "HIDDEN" in exo.name]

NUM_QUESTIONS = sum(len(exo.variations) for exo in EXERCISES)


@dataclass
class Message:
    user: str
    content: str
    timestamp: float = field(default_factory=lambda: time())
    skipped_by_teacher: bool = False


@dataclass
class Question:
    user: str
    exo: int
    variation: int
    messages: list[Message] = field(default_factory=list)
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def needs_response_since(self):
        # Return the timestamp of the first non-teacher message that doesn't have a response and is not skipped
        t = None
        for msg in self.messages[::-1]:
            if msg.user == TEACHER_NAME or msg.skipped_by_teacher:
                return t
            t = msg.timestamp
        return t

    @property
    def never_got_feedback(self):
        return all(msg.user != TEACHER_NAME for msg in self.messages)

    def fmt_messages(self, user: str):
        return "  \n".join(
            f"**{'Me' if msg.user == user else msg.user}**: {msg.content}" for msg in self.messages
        )

    @property
    def last_message_time(self):
        if self.messages:
            return self.messages[-1].timestamp
        return -1

    @classmethod
    def from_dict(cls, data: dict):
        data["messages"] = [Message(**msg) for msg in data["messages"]]
        return cls(**data)

    @property
    def variation_text(self) -> str:
        return EXERCISES[self.exo].variations[self.variation]

    @property
    def full_exo(self) -> Exercise:
        return EXERCISES[self.exo]


class DataBase(dict[str, list[list[Question]]]):
    """
    The database is a dictionary of users, each user has a list of exercises, each exercise has a list of questions.
    """

    def as_json(self) -> dict[str, list[list[dict]]]:
        return {k: [[dataclasses.asdict(q) for q in qs] for qs in v] for k, v in self.items()}

    @classmethod
    def from_json(cls, data: dict[str, list[list[dict]]]) -> "DataBase":
        return cls({k: [[Question.from_dict(q) for q in qs] for qs in v] for k, v in data.items()})

    def reload(self, path: Path) -> None:
        other = DataBase.from_json(json.loads(path.read_text()))
        self.clear()
        self.update(other)
        return self

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_json(), indent=2))

    def login(self, user: str) -> None:
        """Create a new user if it doesn't exist"""
        self.setdefault(
            user,
            [
                [Question(user, e, v) for v in range(len(exo.variations))]
                for e, exo in enumerate(EXERCISES)
            ],
        )

    def questions_needing_feedback(self) -> list[Question]:
        need_response = [q for q in self.all_questions() if q.needs_response_since is not None]
        need_response.sort(key=lambda q: q.needs_response_since)
        return need_response

    def questions_done(self, user: str) -> int:
        return len([q for exo in self[user] for q in exo if q.messages])

    def all_questions(self) -> list[Question]:
        return [q for user in self.values() for qs in user for q in qs]

    def answer_times(self, last_n: int | None = None) -> list[float]:
        """Return the mean time it takes for the teacher to answer a question."""

        answered = [
            q
            for q in self.all_questions()
            if any(message.user == TEACHER_NAME for message in q.messages)
        ]

        # We collect all pairs (user message -> teacher message)
        # With the maximum number of user messages in between
        times = []
        for q in answered:
            sent_at = 0
            for message in q.messages:
                if message.user == TEACHER_NAME:
                    if sent_at != 0:
                        times.append((sent_at, message.timestamp))
                    else:
                        # Two teacher messages in a row
                        pass
                    sent_at = 0
                elif sent_at == 0:
                    sent_at = message.timestamp
                else:
                    # Two user messages in a row
                    pass

        if last_n is not None:
            times.sort()
            times = times[-last_n:]

        return [end - start for start, end in times]

    def mean_answer_time(self, last_n: int | None = None) -> float:
        times = self.answer_times(last_n)
        if times:
            return sum(times) / len(times)
        return float("nan")


@st.cache_resource
def db() -> DataBase:

    # Find latest backup
    try:
        file = max(BACKUP_DIR.iterdir(), key=lambda f: int(f.stem))
    except ValueError:
        return DataBase()

    print(f"Loading backup from {file}")
    try:
        return DataBase().reload(file)
    except Exception as e:
        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥")
        print(e)
        traceback.print_exc()
        return DataBase()


@st.cache_data()
def get_openai_feedback(original: str, submission: str, exo: Exercise, model: str) -> str | None:

    def fmt(orig: str, sub: str):
        return f"Original: {orig}\nResponse: {sub}"

    examples: list[ChatCompletionMessageParam] = []
    for example in exo.examples:
        examples.append({"role": "user", "content": fmt(example.original, example.feedback)})
        examples.append({"role": "assistant", "content": example.feedback})

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": exo.system_prompt},
                *examples,
                {"role": "user", "content": fmt(original, submission)},
            ],
        )
    except Exception as e:
        with st.expander("Silenced error"):
            st.write(e)
            st.exception(e)
        return ""

    return response.choices[0].message.content


def admin_panel():

    db().__class__ = DataBase  # Hack to have the new methods available during development

    with st.sidebar:
        answer_time = db().mean_answer_time()
        st.metric("Mean answer time", f"{answer_time:.0f} seconds")

        model = st.selectbox("OpenAI model", MODELS)

        txt = "## Participants\n"
        if db():
            for u in db():
                txt += f"- **{u}** ({db().questions_done(u)}/{NUM_QUESTIONS})\n"
        else:
            txt += "No participants yet"
        st.markdown(txt)

        # Histogram of answer times

        times = db().answer_times()
        if times:
            df = pd.DataFrame({"time": times})
            hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("time", bin=alt.Bin(maxbins=20), title="Time to answer (s)"),
                    y="count()",
                )
            )
            st.altair_chart(hist, use_container_width=True)

        with st.expander("⚙ Database"):
            st.button("Wipe database", on_click=lambda: db().clear())
            st.download_button(
                "Download current database",
                json.dumps(db().as_json(), indent=2),
                "database.json",
                "Download the database as a JSON file",
            )
            backups = sorted(BACKUP_DIR.iterdir(), reverse=True)
            if backups:
                labeled_backups = {
                    # file: datetime.fromtimestamp(int(file.stem)).strftime("%Y-%m-%d %H:%M:%S")
                    # Tue 1st Jan 2030 00:00:00
                    file: datetime.fromtimestamp(int(file.stem)).strftime("%a %d %b %Y %H:%M:%S")
                    for file in backups
                }
                label = st.selectbox("Backup to load", list(labeled_backups.values()))
                file = next(file for file, date in labeled_backups.items() if date == label)

            else:
                label = None
                st.write("No backups yet")

            new = st.file_uploader("Upload database", type="json")
            if new:
                label = "uploaded file"

            if label is not None:
                # Load a copy of the database
                if label == "uploaded file":
                    new_db = DataBase.from_json(json.loads(new.read().decode()))
                else:
                    new_db = DataBase().reload(file)

                # Show info about the backup
                st.write(
                    f"""
                ### Backup info
                {', '.join(f'**{u}** ({new_db.questions_done(u)}/{NUM_QUESTIONS})' for u in new_db) or 'empty'}
                         """
                )

                if st.button(f"⚠ Load backup from {label}"):
                    db().clear()
                    db().update(new_db)

        # Update the source code code.
        with st.expander("🛠 Source code"):
            modif_timestamp = os.path.getmtime("main.py")
            st.write(f"Last code change: {datetime.fromtimestamp(modif_timestamp):%c}")

            last_commits = os.popen("git log --oneline -n 3").read().strip()
            # Remove the hashes, add bullet points
            last_commits = "\n- ".join(line.partition(" ")[2] for line in last_commits.splitlines())
            st.write(f"Last 3 commits:\n- {last_commits}")

            if st.button("Pull latest version"):
                with st.spinner("running `git pull`..."):
                    output = os.popen("git pull").read()
                st.code(output)
                st.write("Please refresh the page to see the changes, if any.")

        preview_exos = st.toggle("Preview exercises")

    if preview_exos:
        show_openai_prompts = st.checkbox("Show OpenAI prompts")
        for exo in EXERCISES:
            st.write(exo.instructions)
            for i, v in enumerate(exo.variations):
                st.markdown(f"### Variation {i+1}\n{v}")
                st.text_input("Answer", disabled=True, key=v)

                if show_openai_prompts:
                    st.write("System prompt")
                    st.code(exo.system_prompt)
                    for example in exo.examples:
                        st.write("Example")
                        st.code(f"Original: {example.original}\nRephrase: {example.student}")
                        st.write("Feedback")
                        st.code(example.feedback)
            st.divider()

    st.write("# Feedback panel")

    for q in db().questions_needing_feedback():
        exo = EXERCISES[q.exo]
        variation = exo.variations[q.variation]

        # This container helps clearing the screen faster.
        # We cont.empty() when we're done with the feedback
        # Which avoid having streamlit remove the elements one by one
        # and gains ~1.5s per question
        cont = st.container()
        with cont:
            st.markdown(f"## **{q.user}** on {exo.name}  \n{variation}")

            with st.expander("Context"):
                st.write(exo.instructions)
                st.divider()

                # Collect last 5 questions with discussions
                qs = sorted(
                    [q_ for exo in db()[q.user] for q_ in exo if q_.messages and q_ is not q],
                    key=lambda q: q.last_message_time,
                    reverse=True,
                )[:5]

                if qs:
                    st.write("#### Previous chats")
                    for q_ in qs:
                        st.write(f"On **{q_.full_exo.name}** — {q_.variation_text}")
                        st.write(q_.fmt_messages(TEACHER_NAME))
                else:
                    st.write("No previous chats")

            st.write(q.fmt_messages(TEACHER_NAME))

            if q.never_got_feedback and model:
                default = get_openai_feedback(
                    variation, q.messages[0].content, EXERCISES[q.exo], model
                )
            elif q.never_got_feedback:
                default = ""
            else:
                default = ""

            with st.form(key=f"form-{q.uid}"):
                new_msg = st.text_area("Feedback", value=default, height=250, key=q.uid)
                submit = st.form_submit_button("Send")
            if new_msg and submit:
                q.messages.append(Message(TEACHER_NAME, new_msg))
                cont.empty()
                st.rerun()

            # Allow to skip a message if any feedback has been sent.
            # The condition is important to avoid softlocks. The participants can't continue if the teacher doesn't send feedback.
            if not q.never_got_feedback:
                skip = st.button("Skip", key="skip" + q.uid)
                if skip:
                    q.messages[-1].skipped_by_teacher = True
                    cont.empty()
                    st.rerun()

    # Check for new questions every second
    old_db = deepcopy(db())
    while True:
        if old_db != db():
            # Backup the database every new message
            db().save(BACKUP_DIR / f"{int(time())}.json")
            st.rerun()
        sleep(0.5)


def main():
    user = st.session_state.get("user")

    if user is None:
        # Prompt for user name
        user = st.text_input("Your name")
        if not user:
            return
        st.session_state.user = user
        st.rerun()  # So we remove the text input
    elif user == TEACHER_NAME:
        admin_panel()
        return
    else:
        st.write(f"Welcome {user}!")

    # Create a new user if it doesn't exist
    db().login(user)

    for i, (exo, qs) in enumerate(zip(EXERCISES, db()[user])):
        anchor = f"<a name='exo-{i+1}'></a>\n"
        st.write(anchor + exo.instructions, unsafe_allow_html=True)

        for i, (q, variation) in enumerate(zip(qs, exo.variations)):
            st.markdown(f"### Variation {i+1}\n{variation}")

            with st.container():
                st.write(q.fmt_messages(user))
                if len(q.messages) != 1 and (new := st.chat_input(key=f"chat-{q.uid}")):
                    q.messages.append(Message(user, new))
                    st.rerun()

            # If there was never any feedback, don't show the following questions
            if user == "D":
                continue
            if q.never_got_feedback:
                break
        else:
            st.divider()
            continue
        break

    with st.sidebar:
        st.write("*Table of contents*")
        toc = ""
        for i, (exo, qs) in enumerate(zip(EXERCISES, db()[user])):
            # First line with a # is the title
            header = next(
                (line for line in exo.instructions.splitlines() if line.strip().startswith("#")),
                "no header",
            )
            title = header.lstrip("# ")
            depth = header[: -len(title)].count("#")
            # if title != "no header":
            if depth == 1:
                toc += f"\n[{title}](#exo-{i+1})  "
            elif depth == 2:
                toc += f"- [{title}](#exo-{i+1})  "
            toc += "\n"
            # st.write(f"[{title}](#exo-{i+1})")
            # if any(q.never_got_feedback for q in qs):
            # break

        st.markdown(toc)

        st.metric("Progress", f"{db().questions_done(user)}/{NUM_QUESTIONS}")

    with st.sidebar:
        timer = st.empty()
    last_snow = 0

    # Check for new messages every second
    past_msgs = deepcopy(db()[user])
    while True:
        if past_msgs != db()[user]:
            st.rerun()
        sleep(0.5)

        # Show the timer for the current question
        done = [q for exo in db()[user] for q in exo if q.messages]
        to_do = [q for exo in db()[user] for q in exo if not q.messages]

        if not done or not to_do or to_do[0].full_exo.disable_timer:
            continue

        time_since_last_msg = time() - done[-1].last_message_time
        time_left_for_question = TIME_PER_QUESTION - time_since_last_msg
        minutes, seconds = divmod(abs(time_left_for_question), 60)
        if time_left_for_question >= 0:
            timer.metric("Time left for the question", f"{minutes:02.0f}:{seconds:02.0f}")
        else:
            timer.metric("Time over since", f"{minutes:02.0f}:{seconds:02.0f}")

            if time() - last_snow > 40:
                last_snow = time()
                st.snow()
                st.toast("Time's up, try to submit :)")


if __name__ == "__main__":
    main()
