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


@st.cache_resource
def db() -> dict[str, list[list[Question]]]:
    # Find latest backup
    file = max(BACKUP_DIR.iterdir(), key=lambda f: int(f.stem), default=None)
    if file:
        print(f"Loading backup from {file}")
        try:
            return db_from_json(json.loads(file.read_text()))
        except Exception as e:
            print("🔥🔥🔥🔥🔥🔥🔥🔥🔥")
            print(e)
            traceback.print_exc()
            return {}
    return {}


def db_as_json() -> dict[str, list[list[dict]]]:
    return {k: [[dataclasses.asdict(q) for q in qs] for qs in v] for k, v in db().items()}


def db_from_json(data: dict[str, list[list[dict]]]):
    return {k: [[Question.from_dict(q) for q in qs] for qs in v] for k, v in data.items()}


def wait_feedback(question: Question):
    assert question.messages

    with st.spinner("Getting feedback..."):
        while question.messages[-1].user is not TEACHER_NAME:
            sleep(1)


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
    model = st.selectbox("OpenAI model", MODELS)

    with st.expander("Database"):
        st.button("Wipe database", on_click=lambda: db().clear())
        backups = sorted(BACKUP_DIR.iterdir(), reverse=True)
        if backups:
            labeled_backups = {
                file: datetime.fromtimestamp(int(file.stem)).strftime("%Y-%m-%d %H:%M:%S")
                for file in backups
            }
            label = st.selectbox("Database to load", list(labeled_backups.values()))
            file = next(file for file, date in labeled_backups.items() if date == label)

            if st.button(f"⚠ Load backup from {label}"):
                db().clear()
                db().update(db_from_json(json.loads(file.read_text())))
        else:
            st.write("No backups yet")
        st.write(db())

    with st.expander("Preview exercises"):
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

    need_response = [
        q for user in db().values() for qs in user for q in qs if q.needs_response_since
    ]
    need_response.sort(key=lambda q: q.needs_response_since)

    for q in need_response:
        exo = EXERCISES[q.exo]
        variation = exo.variations[q.variation]
        st.markdown(f"## **{q.user}** on {variation}")
        with st.expander("Context"):
            st.write("Exercise:")
            st.write(exo.instructions)
            st.divider()

            # Collect last 5 questions
            qs = sorted(
                [q_ for exo in db()[q.user] for q_ in exo if q_.messages and q_ is not q],
                key=lambda q: q.last_message_time,
                reverse=True,
            )[:5]

            if qs:
                st.write("Previous chats:")
                for q_ in qs:
                    st.write(q_.variation)
                    st.write(q_.fmt_messages(TEACHER_NAME))
            else:
                st.write("No previous chats")

        st.write(q.fmt_messages(TEACHER_NAME))

        if q.never_got_feedback and model:
            default = get_openai_feedback(variation, q.messages[0].content, EXERCISES[q.exo], model)
        elif q.never_got_feedback:
            default = ""
        else:
            default = ""

        with st.form(key=f"form-{q.uid}"):
            new_msg = st.text_area("Feedback", value=default, height=250, key=q.uid)
            submit = st.form_submit_button("Send")
        if new_msg and submit:
            q.messages.append(Message(TEACHER_NAME, new_msg))
            st.rerun()

        # Allow to skip a message if any feedback has been sent.
        # The condition is important to avoid softlocks. The participants can't continue if the teacher doesn't send feedback.
        if not q.never_got_feedback:
            skip = st.button("Skip", key="skip" + q.uid)
            if skip:
                q.messages[-1].skipped_by_teacher = True
                st.rerun()

    # Check for new questions every second
    old_db = deepcopy(db())
    while True:
        if old_db != db():
            # Backup the database every new message
            (BACKUP_DIR / f"{int(time())}.json").write_text(json.dumps(db_as_json(), indent=2))
            st.rerun()
        sleep(1)


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
        st.write(f"Welcome {user}")

    # Create a new user if it doesn't exist
    db().setdefault(
        user,
        [
            [Question(user, e, i) for i in range(len(exo.variations))]
            for e, exo in enumerate(EXERCISES)
        ],
    )

    for i, (exo, qs) in enumerate(zip(EXERCISES, db()[user])):
        anchor = f"<a name='exo-{i+1}'></a>"
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

        number_of_questions = sum(len(exo.variations) for exo in EXERCISES)
        questions_done = len([q for exo in db()[user] for q in exo if q.messages])
        st.metric("Progress", f"{questions_done}/{number_of_questions}")

    with st.sidebar:
        timer = st.empty()
    last_snow = 0

    # Check for new messages every second
    past_msgs = deepcopy(db()[user])
    while True:
        if past_msgs != db()[user]:
            st.rerun()
        sleep(0.5)

        # Show the timer
        try:
            last_msg = max(q.last_message_time for exo in db()[user] for q in exo if q.messages)
        except ValueError:  # No messages
            pass
        else:
            time_since_last_msg = time() - last_msg
            time_left_for_question = TIME_PER_QUESTION - time_since_last_msg
            if time_left_for_question > 0:
                minutes, seconds = divmod(time_left_for_question, 60)
                timer.metric("Time left for the question", f"{minutes:02.0f}:{seconds:02.0f}")
            else:
                minutes, seconds = divmod(-time_left_for_question, 60)
                timer.metric("Time left for the question", f"-{minutes:02.0f}:{seconds:02.0f}")

                if time() - last_snow > 40:
                    last_snow = time()
                    st.snow()
                    st.toast("Time's up, try to submit :)")


if __name__ == "__main__":
    main()
