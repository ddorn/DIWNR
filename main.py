from copy import deepcopy
from dataclasses import field, dataclass
from time import sleep, time
import uuid
import streamlit as st
import openai
from openai.types.chat import ChatCompletionMessageParam


# MODEL="gpt-4-0125-preview"
MODEL = "gpt-3.5-turbo-0125"
TEACHER_NAME = "Camille"


@dataclass
class Exercise:
    instructions: str
    """Initial instructions shown at the start of the exercise"""
    variations: list[str]
    """Variations using the same instructions"""
    system_prompt: str
    """System prompt given to gpt for feedback"""
    examples: list[tuple[str, str, str]]
    """Examples of triplets (question, answer, feedback), to guide gpt-4"""


EXO_1 = Exercise(
    instructions="""
**For all the following exercices, you’re invited to always put yourself in the shoes of the Questioner. You’ll need to give an answer to something the interlocutor (IL) has said, while applying the tip (⚙️) you're given. Pretend to be in a *live conversation*.**

# 1 - 🎯 Optimizing Rapport Management
Rapport is fundamental for having those discussions -it influences trust, the interpretation of what you're saying, and the ability to coherentize one's belief system.

⚙️ A good way to improve rapport is simply **rephrasing what your interlocutor has just said**. Make sure to include the **emotions** they mention, and to ask  **whether you got it right**. (e.g “don’t hesitate to tell me if I’m wrong). Try to stick to what they said, do not presume too much.

*This allows to make sure that you've understood them. It also forces you to pay more attention to what they're saying. Finally, it also avoids to incorrectly "ground": to signal that your interlocutor can proceed to the next point, despite the fact that you didn't understand.*

""",
    variations=[
        """ IL:\"I don't see why animal suffering matters, animals aren't intelligent.\"
        Inspiring yourself from the tip (⚙️) above, rephrase what they're saying.""",
        """IL:\"I think we shouldn't be afraid of AI.\"
      Inspiring yourself from the tip (⚙️) above, rephrase what they're saying.""",
    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
You’re invited to always put yourself in the shoes of the Questioner. You’ll need to give an answer to something the interlocutor (IL) has said

Rapport is fundamental for having those discussions -it influences trust, the interpretation of what you're saying, and the ability to coherentize one's belief system.
⚙️ A good way to improve rapport is simply rephrasing what your interlocutor has just said. Make sure to include the emotions they mention, and to ask whether you got it right. (e.g “don’t hesitate to tell me if I’m wrong). Try to stick to what they said, do not presume too much.
This allows to make sure that you've understood them. It also forces you to pay more attention to what they're saying. Finally, it also avoids to incorrectly "ground": to signal that your interlocutor can proceed to the next point, despite the fact that you didn't understand.
----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("I don't see why AI would be dangerous, it's not that powerful.",
         "AI can’t do enough to really cause us any problems we can’t solve.",
         "Good, make sure to check whether you’ve understood your interlocutor (e.g;, “did I understand you well ?”)"),
        ("I don't see why AI would be dangerous, it's not that powerful.",
         "I see. I presume by “powerful”, you mean that it can’t actually affect things in the world, and since being dangerous means creating dangerous situations, AI can’t do this since it’s essentially a fancy computer program. Is that a fair restatement of your views?",
         "Very Good! Try to be more concise.")
    ]
)


EXO_2 = Exercise(
    instructions="""
Sometimes, the rapport might imply someone who identifies you as being part of the “other side”.

⚙️ A way to counter this is to **find values that you share** with your interlocutor. l call this **"terminal affiliation"**.

*Typically, values and terminal goals can be agreed upon. E.g.
A: "I don't want to regulate AI because I want to protect the market and human innovation"
B: "I agree that we should protect human flourishing and innovation".
Typically, facts or parts of one’s beliefs are harder to agree on, and even so might sound misleading (e.g. “I agree that it is unclear who killed Kennedy” might imply in your interlocutor’s mind that you should therefore agree with them that person X killed Kennedy).

This allows to break the erroneous outgroup depiction one person might have. If you feel there is a risk of misrepresentation, you can also mention things that you don't agree with, although in contexts where you use this technique, a lot of disagreement is already anticipated.*
""",
    variations=[
        """IL:"I hate progressives because they are threatening the safety of our institutions."
        Offer terminal affiliation.
""",
        """ IL:"Climate Change is fake, it's a lie that's been put out there by the global elite".
        Offer terminal affiliation.
""",
        """ IL: I can't stand this whole lab meat research. My health is more important than their profit!
        Offer terminal affiliation"""
    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
Sometimes, the rapport might imply someone who identifies you as being part of the “other side”.
⚙️ A way to counter this is to find values that you share with your interlocutor. l call this "terminal affiliation".
Typically, values and terminal goals can be agreed upon. E.g.
A: I don't want to regulate AI because I want to protect the market and human innovation"
B: "I agree that we should protect human flourishing and innovation".
Typically, facts or parts of one’s beliefs are harder to agree on, and even so might sound misleading (e.g. “I agree that it is unclear who killed Kennedy” might imply in your interlocutor’s mind that you should therefore agree with them that person X killed Kennedy).

This allows to break the erroneous outgroup depiction one person might have. If you feel there is a risk of misrepresentation, you can also mention things that you don't agree with, although in contexts where you use this technique, a lot of disagreement is already anticipated.
----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("I hate progressives because they are threatening the safety of our institutions.",
         "I certainly want to maintain the safety of  our institutions, not tear them down!.",
         "Great, note that it might sound slightly defensive. You could insist by saying I want to maintain the safety of our institutions too)."),
        ("I hate progressives because they are threatening the safety of our institutions.",
         "I sympathize! Societal institutions are definitely important for a well-functioning society, and I don’t like them being threatened, either.",
         "Excellent!")
    ]
)


EXO_3 = Exercise(
    instructions="""
⚙️Just a side remark, you might want to **sparkle a few "Good", "Cool", "Ok", "I see", "Mhm"**, as your interlocutor is talking, and also at the start of your turn. This is called an **encourager**.

*It signals that you've paid attention to what they've been saying, and that you “close” your speaking turn before starting a new one. This is crucial for maintaining good rapport.
Note however that encouragers are not affiliation markers, they don't necessarily signal that you agree ! Rather, they signal that you "converge", that you jointly pay attention to what they pay attention to. In some languages, the encourager for "yes, I follow you" and the affiliation marker for "yes, I agree" are two different words.*

""",
    variations=[
        """IL:"I hate conservatives because they're just racist."
        Rephrase with an encourager.
""",
        """IL:"I don't think AI will become superintelligent this century, techies are just overhyped".
        Offer terminal affiliation.
""",
    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
⚙️Just a side remark, you might want to sparkle a few "Good", "Cool", "Ok", "I see", "Mhm", as your interlocutor is talking, and also at the start of your turn. This is called an encourager.
It signals that you've paid attention to what they've been saying, and that you “close” your speaking turn before starting a new one. This is crucial for maintaining good rapport.
Note however that encouragers are not affiliation markers, they don't necessarily signal that you agree ! Rather, they signal that you "converge", that you jointly pay attention to what they pay attention to. In some languages, the encourager for "yes, I follow you" and the affiliation marker for "yes, I agree" are two different words.

----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("I hate conservatives because they're just racist.",
         "Huh. I see, go on",
         "Good, be sure to also rephrase what they're saying in the process."),
        ("I hate conservatives because they're just racist.",
         "Mmm, understood. Their views feel unfair to you",
         "Excellent!")
    ]
)

EXO_4 = Exercise(
    instructions="""
⚙️Enumerating helps to "converge" on the cognitive level, to validate the fact that you got their point across. It's usually done by means of "Not only do you... but you also..." or "Not only because... but also because..", etc.

""",
    variations=[
        """IL:"I can't stand those so-called effective charities, they don't care about economic development, they don't care about the local culture of the population they intervene on, and, like, where's my responsibility in giving them money?"
Rephrase with enumerating.

""",
        """IL:"I'm depressed with all those regulations on fishing, it's  time lost for me and also insulting -I know my quotas, I don't need someone to check if I respect them."
        Rephrase with enumerating.
""",
    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
⚙️Enumerating helps to "converge" on the cognitive level, to validate the fact that you got their point across. It's usually done by means of "Not only do you... but you also..." or "Not only because... but also because..", etc.

----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("I can't stand those so-called effective charities, they don't care about economic development, they don't care about the local culture of the population they intervene on, and, like, where's my responsibility in giving them money?",
         "So you feel the experts are not taking businesses into consideration, they are just trying to keep development from happening",
         "Not really. In general, it’s a bit like “doing a bullet point list” of what you’re interlocutor has just said, instead of summarizing it.)."),
        ("I can't stand those so-called effective charities, they don't care about economic development, they don't care about the local culture of the population they intervene on, and, like, where's my responsibility in giving them money?",
         "I see, not only do you think these charities fail at adressing a bigger problem, they also neglect human factors, and moreover you feel they demand a responsibility that you don't think you have.",
         "Excellent!")
    ]
)

EXO_5 = Exercise(
    instructions="""
 ⚙️ Including the nuances your interlocutor mention are also very welcome. E.g: “On one side, you think that X, but on the other side, you also think that Y.”

""",
    variations=[
        """IL:"I'm not saying we should make abortion illegal, but abortion supporters are pushing it too far. We should raise responsible citizens."
        Rephrase with nuanced contrasting.

""",
        """IL:"Vegans are not idiots, they're just irritating and excessively agressive, but I guess we could have a conversation if they behaved better."
Rephrase with nuances.

""",
    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
⚙️ Including the nuances your interlocutor mention are also very welcome. E.g: “On one side, you think that X, but on the other side, you also think that Y.”

----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"I'm not saying we should make abortion illegal, but abortion supporters are pushing it too far. We should raise responsible citizens."
        Rephrase with nuanced contrasting.""",
         "So like, you think a certain amount of restriction makes sense, but not prohibit it entirely?",
         "Good!"),
        ("""IL:"I'm not saying we should make abortion illegal, but abortion supporters are pushing it too far. We should raise responsible citizens."
        Rephrase with nuanced contrasting.""",
         "Right, so on the one hand,  you think more regulations are needed, but on the other, you  don't think we should prohibit it entirely.",
         "Excellent!")
    ]
)

EXO_6 = Exercise(
    instructions="""
 Congrats! You finished section 1.
 # 2 - 📖 Optimizing Narrative Transportation

 To access section 2, please open the following link in a new window: https://www.guidedtrack.com/programs/kwbiasj/run
 Please do not close this window, simply get back to here once you're ended, and send "Done" in the field text below. ”

""",
    variations=[
        """


""",

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
send "Done" in the field text below.

----
Your task as an assistant is to answer with "Good".
""",
    examples=[
        ("""send "Done" in the field text below.""",
         "Done",
         "Good!"),
        ("""send "Done" in the field text below.""",
         "Done",
         "Excellent!")
    ]
)

EXO_7 = Exercise(
    instructions="""
Congrats! You finished section 2.
# 3 - 🎯 Socratic Tests

Tests point to the logical relations between the different sentences that your interlocutor said. Due to the fact that interlocutors are more convinced by the inferences they draw themselves rather than the information they're being given directly, your goal is to help them make the right inferences -i.e logical or bayesian ones- by relying on elements they themselves introduced. Tests are “transformations” of what your interlocutor said.

This is **not** something questioners usually improvise on the fly. Most of the time, they draw from a "library" of tests depending on what seems most fitting. I'll try to teach you a few tests.

 ⚙️ First is the **Outsider Test**. This technique is good for beliefs that are usually attached to a "Me" or "Us". It consists in finding an similar case where someone would come to a different conclusion by relying on the same rationale.

Since tests can come off as a little upfront, you might want to add question verbs such as “I wonder whether” or “Could we say that…” to soothe it out.

  ”

""",
    variations=[
        """IL:"I believe in the christian God because my parents are christian."
Apply the Outsider Test.
""",

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
Tests point to the logical relations between the different sentences that your interlocutor said. Due to the fact that interlocutors are more convinced by the inferences they draw themselves rather than the information they're being given directly, your goal is to help them make the right inferences -i.e logical or bayesian ones- by relying on elements they themselves introduced. Tests are “transformations” of what your interlocutor said.

This is not something questioners usually improvise on the fly. Most of the time, they draw from a "library" of tests depending on what seems most fitting. I'll try to teach you a few tests.

 ⚙️ First is the Outsider Test. This technique is good for beliefs that are usually attached to a "Me" or "Us". It consists in finding an similar case where someone would come to a different conclusion by relying on the same rationale.

 Since tests can come off as a little upfront, you might want to add question verbs such as “I wonder whether” or “Could we say that…” to soothe it out.

----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"I believe in the christian God because my parents are christian."
Apply the Outsider Test.
""",
         "Huh, I wonder if someone with Hindu parents would say the same thing, though?",
         "Excellent!"),
        ("""IL:"I believe in the christian God because my parents are christian."
Apply the Outsider Test.""",
         "If that's true then you should believe in Vishnu",
         "Not really. Make sure to go step-by-step, e.g. asking what it would have been like if they were born in a Hindu family. Also, you can use a question verb (e.g. I wonder if, if that's true) to make it sound nicer.")
    ]
)

EXO_8 = Exercise(
    instructions="""
⚙️Another test is the Moorean Shift. It simply consists of exploiting the logical equivalence A -> B => ~B -> ~A. This also induces a topic-shift.

E.g (courtesy of EY) :
*"Only God can build an A.I (this implies : If you are not God, you cannot build an AI)
-If I build an AI, does that mean I'm God ?"*
Notice the conversations switches from being about AI onto being about God, so make sure the result does orient you towards the topic you aim for.
Sometimes, sticking to the exact same topic is a bit tricky. An alternative is to select a broader category:
*-If I build something more advanced than a human, does that mean I'm superhuman?*
  ”

""",
    variations=[
        """IL:"Only aliens could build the pyramids, because they are so huge."
Question your interlocutor with a Moorean Shift.

""",

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
Tests point to the logical relations between the different sentences that your interlocutor said. Due to the fact that interlocutors are more convinced by the inferences they draw themselves rather than the information they're being given directly, your goal is to help them make the right inferences -i.e logical or bayesian ones- by relying on elements they themselves introduced. Tests are “transformations” of what your interlocutor said.

⚙️Another test is the Moorean Shift. It simply consists of exploiting the logical equivalence A -> B => ~B -> ~A. This also induces a topic-shift.

E.g (courtesy of EY) :
"Only God can build an A.I (this implies : If you are not God, you cannot build an AI)
-If I build an AI, does that mean I'm God ?"
Notice the conversations switches from being about AI onto being about God, so make sure the result does orient you towards the topic you aim for.
Sometimes, sticking to the exact same topic is a bit tricky. An alternative is to select a broader category:
-If I build something more advanced than a human, does that mean I'm superhuman?

----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"Only aliens could build the pyramids, because they are so huge."
Question your interlocutor with a Moorean Shift.
""",
         "If something is over a certain size, does it mean it has been built by aliens?",
         "Excellent!"),
        ("""IL:"Only aliens could build the pyramids, because they are so huge."
Question your interlocutor with a Moorean Shift.""",
         "If I build a pyramid, does that mean I'm an alien?",
         "Nearly there!  The idea of you building a pyramid is a bit far-fetched, but you got the idea. You can rephrase it by using a broader category, e.g. If someone builds something very large,  does it mean they're alien?")
    ]
)

EXO_9 = Exercise(
    instructions="""
⚙️Modal Breaching ! Someone might use a weak modal, either explicitly ("I can", "one may") or implicitly ("I guess", "I think you would"). They point to a possible world that they still haven't explored yet. This consists in exploring the case where the modal is not verified.

*E.g. :
"I'm definitely my mother's son, because, I imagine that if I make a DNA test then it'll just match my parents’".
"Let's suppose you make the DNA test and it doesn't match. Does that influence your belief ?"*

  ”

""",
    variations=[
        """IL:"I believe my father is 93 because... I guess he has a birth certificate..."
Make a Modal Breach.


""",

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
Tests point to the logical relations between the different sentences that your interlocutor said. Due to the fact that interlocutors are more convinced by the inferences they draw themselves rather than the information they're being given directly, your goal is to help them make the right inferences -i.e logical or bayesian ones- by relying on elements they themselves introduced. Tests are “transformations” of what your interlocutor said.

⚙️Modal Breaching ! Someone might use a weak modal, either explicitly ("I can", "one may") or implicitly ("I guess", "I think you would"). They point to a possible world that they still haven't explored yet. This consists in exploring the case where the modal is not verified.

E.g. :
"I'm definitely my mother's son, because, I imagine that if I make a DNA test then it'll just match my parents’".
"Let's suppose you make the DNA test and it doesn't match. Does that influence your belief ?"
----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"I believe my father is 93 because... I guess he has a birth certificate..."
Make a Modal Breach.
""",
         "What if his birth certificate says something different, or record keeping in his country’s inaccurate? Would you still think he’s 93?",
         "Excellent!"),
        ("""IL:"I believe my father is 93 because... I guess he has a birth certificate..."
Make a Modal Breach.
""",
         "But you can't be sure, right?",
         "Not quite, the idea is to ask what would happen if the certificate e.g. isn't found or says something different.")
    ]
)

EXO_10 = Exercise(
    instructions="""
⚙️**Transfers** are relatively simple. Psychologists usually categorize reasoning into different uses, called prospective and retrospective.
*Prospective reasoning
**inquiry
**argumentation
**prediction
*Retrospective reasoning
**explanation
**justification

Descriptively speaking, these uses are debatable. However, they offer you a good test. You can indeed test if assertions in one use transfer to the other use, e.g: IL:”Immigrants want to replace us” Q:”So, if I give you the name of someone who does not want to replace us, then you’ll be able to systematically guess that it’s not an immigrant ?”

You don't need to limit yourself to these uses, of course. Most of the time, you depart from argumentation, and want to see if the reason used in another context still holds water (e.g, from argumentation to prediction, or from argumentation to explanation)

  ”

""",
    variations=[
        """IL:"All conservatives want to implement racist laws."
Make a Transfer.



""",

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
⚙️Transfers are relatively simple. Psychologists usually categorize reasoning into different uses, called prospective and retrospective.
-Prospective reasoning. It includes inquiry, argumentation, prediction.
-Retrospective reasoning. It includes explanation and justification

Descriptively speaking, these uses are debatable. However, they offer you a good test. You can indeed test if assertions in one use transfer to the other use, e.g: IL:”Immigrants want to replace us” Q:”So, if I give you the name of someone who does not want to replace us, then you’ll be able to systematically guess that it’s not an immigrant ?”

You don't need to limit yourself to these uses, of course. Most of the time, you depart from argumentation, and want to see if the reason used in another context still holds water (e.g, from argumentation to prediction, or from argumentation to explanation)
----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"All conservatives want to implement racist laws."
Make a Transfer.
""",
         "If I introduce you to someone who is against racist laws, you’d be able to infer they’re not conservative?",
         "Excellent!"),
        ("""IL:"All conservatives want to implement racist laws."
Make a Transfer.
""",
         "So does that mean the Soviets (who implemented racist laws) were conservative?",
         "Not ideal. You're introducing a new element for your interlocutor, and they might disagree with it. It's better to probe for their own example instead.")
    ]
)

EXO_11 = Exercise(
    instructions="""
⚙️**Testing false positives** is a useful technique when the interlocutor seems to display confirmation bias.

E.g :
"My patients come to see me to realign their chakra, and afterward, they are healed.
"-Do you know of someone who was healed yet didn't come to see you?"
  ”

""",
    variations=[
        """IL:"I believe eating berries in the morning makes you healthy because, when I do, I'm always between 50 and 53 kg."
Ask for a false positive.
""",

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
Tests point to the logical relations between the different sentences that your interlocutor said. Due to the fact that interlocutors are more convinced by the inferences they draw themselves rather than the information they're being given directly, your goal is to help them make the right inferences -i.e logical or bayesian ones- by relying on elements they themselves introduced. Tests are “transformations” of what your interlocutor said.

⚙️Testing false positives is a useful technique when the interlocutor seems to display confirmation bias.

E.g :
"My patients come to see me to realign their chakra, and afterward, they are healed.
"-Do you know of someone who was healed yet didn't come to see you?"
----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"I believe eating berries in the morning makes you healthy because, when I do, I'm always between 50 and 53 kg."
Ask for a false positive.
""",
         "Can you think of a morning where you didn’t eat berries, and yet you weighed between 50 and 53 kgs?",
         "Excellent!"),
        ("""IL:"I believe eating berries in the morning makes you healthy because, when I do, I'm always between 50 and 53 kg."
Ask for a false positive.
""",
         "Do you remember a time when you ate berries and were not in that weight range?",
         "Not really, the interlocutor already said she only remember being in this weight range. Try probing for a time when they did NOT eat berries yet were in this weight range.")
    ]
)

EXO_12 = Exercise(
    instructions="""
#5 - 🎯 Managing Topics
Topics are the structure of your interlocutor's belief woven into conversation. Any of these topics usually have several versions of them, so you might ask for the "strongest", "most important", "main", or "most vivid" ones at each conversational step.

Here is a way (among other) to describe the typical structure you’ll encounter :
(image)
C refers to the Claim : “I believe in God”. This is when you usually try to clear up definitions.

D is the Data : “I had a spiritual experience, once”. You often need to ask for an example at this point.

W is the Warrant, the link that justified inferring the Claim from the Data :
E.g. : “Spiritual experiences reveal deeper truth than science or logic”.

R is the Rebuttal : “Unless if the spiritual experience was induced by a drug or something but I don’t think it was the case”.

B is the Backing
Backing can exist for both Data (“I was 13. I was praying with my aunt in a church…”) and the Warrant (“I actually read a book titled “The book of the Shaman…”)

Most importantly, we want two things :
1-Get to the right Data. Data sometimes has another Data behind it, and so on and so forth. You want to get to a concrete Data.
2-Once you have the right data, *Get to the Warrant and stick to it !* This is where the interlocutor’s epistemology happens, so be wary of falling inside unrelated Claims, optional Data, or unnecessarily detailed Backings.
""",
    variations=[
        """IL:"Cool, I'm open for a discussion."
Ask for a claim.
""",
      """IL:"I believe in Karma."
Ask for Data.
      """,
      """IL:"I believe in Karma because I just experience it everyday."
Ask for an example.
      """,
      """IL:"Like, yesterday, I mentally insulted someone, and a few minutes later, my foot hit a table corner."
Ask for a Warrant… in an appropriate way.
      """,
      """IL:"I believe this incident was Karma."
Ask for a rebuttal.
      """,

    ],
    system_prompt="""
Participants are asked to complete the following exercise:
---
Topics are the structure of your interlocutor's belief woven into conversation. Any of these topics usually have several versions of them, so you might ask for the "strongest", "most important", "main", or "most vivid" ones at each conversational step.

Here is a way (among other) to describe the typical structure you’ll encounter :
(image)
C refers to the Claim : “I believe in God”. This is when you usually try to clear up definitions.

D is the Data : “I had a spiritual experience, once”. You often need to ask for an example at this point.

W is the Warrant, the link that justified inferring the Claim from the Data :
E.g. : “Spiritual experiences reveal deeper truth than science or logic”.

R is the Rebuttal : “Unless if the spiritual experience was induced by a drug or something but I don’t think it was the case”.

B is the Backing
Backing can exist for both Data (“I was 13. I was praying with my aunt in a church…”) and the Warrant (“I actually read a book titled “The book of the Shaman…”)

Most importantly, we want two things :
1-Get to the right Data. Data sometimes has another Data behind it, and so on and so forth. You want to get to a concrete Data.
2-Once you have the right data, Get to the Warrant and stick to it !This is where the interlocutor’s epistemology happens, so be wary of falling inside unrelated Claims, optional Data, or unnecessarily detailed Backings.
----
Your task as an assistant is to provide them with feedback to improve on their rephrasing.
""",
    examples=[
        ("""IL:"Cool, I'm open for a discussion."
Ask for a claim.
""",
         "Nice! Is there any belief that you hold that you'd want to talk about?",
         "Excellent!"),
        ("""IL:"I believe in Karma."
Ask for Data.
""",
         "Why do you believe in Karma?",
         "Good, but imprecise: try to ask for the main reason why they believe in Karma"),
        ("""IL:"I believe in Karma."
Ask for Data.
""",
          "What do you mean by Karma?",
          "Clarifying definitions is an important step, but this isn't Data. Ask what is the main reason why they believe in Karma.")
    ]
)

EXERCISES = [EXO_1, EXO_2, EXO_3, EXO_4, EXO_5, EXO_6, EXO_7, EXO_8, EXO_9, EXO_10, EXO_11, EXO_12]


@dataclass
class Message:
    user: str
    content: str
    timestamp: float = field(default_factory=lambda: time())


@dataclass
class Question:
    user: str
    original: str
    exo: Exercise
    messages: list[Message] = field(default_factory=list)
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def needs_response(self):
        return self.messages and self.messages[-1].user is not TEACHER_NAME

    @property
    def needs_response_since(self):
        # Return the timestamp of the first non-teacher message that doesn't have a response
        t = None
        for msg in self.messages[::-1]:
            if msg.user == TEACHER_NAME:
                return t
            t = msg.timestamp
        return t

    @property
    def never_got_feedback(self):
        return all(msg.user != TEACHER_NAME for msg in self.messages)

    def fmt_messages(self, user: str):
        return "  \n".join(
            f"**{'Me' if msg.user == user else msg.user}**: {msg.content}"
            for msg in self.messages
        )

    # Don't show the exercise in __repr__
    def __repr__(self):
        return f"Question(user={self.user}, original={self.original}, messages={self.messages})"

    __str__ = __repr__


@st.cache_resource
def db() -> dict[str, list[list[Question]]]:
    return {}


def wait_feedback(question: Question):
    assert question.messages

    with st.spinner("Getting feedback..."):
        while question.messages[-1].user is not TEACHER_NAME:
            sleep(1)


@st.cache_data()
def get_openai_feedback(original: str, submission: str, exo: Exercise) -> str | None:

    def fmt(orig: str, sub: str):
        return f"Original: {orig}\nRephrase: {sub}"

    examples : list[ChatCompletionMessageParam]= []
    for orig, sub, feedback in exo.examples:
        examples.append({"role": "user", "content": fmt(orig, sub)})
        examples.append({"role": "assistant", "content": feedback})

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": exo.system_prompt},
            *examples,
            {"role": "user", "content": fmt(original, submission)},
        ],
    )

    return response.choices[0].message.content


def admin_panel():
    st.button("Wipe database", on_click=lambda: db().clear())
    use_openai = st.toggle("Use OpenAI", False)
    with st.expander("Database"):
        st.write(db())

    with st.expander("Preview exercises"):
        for exo in EXERCISES:
            st.write(exo.instructions)
            for i, v in enumerate(exo.variations):
                st.markdown(f"## {i+1}. {v}")
                st.text_input("Answer", disabled=True, key=v)

                st.write("System prompt")
                st.code(exo.system_prompt)
                for orig, sub, feedback in exo.examples:
                    st.write("Example")
                    st.code(f"Original: {orig}\nRephrase: {sub}")
                    st.write("Feedback")
                    st.code(feedback)


    st.write("# Feedback panel")

    need_response = [q for user in db().values() for qs in user for q in qs if q.needs_response]
    need_response.sort(key=lambda q: q.needs_response_since)

    for q in need_response:
        st.markdown(f"""
        ## **{q.user}** on "{q.original}"
""")
        st.write(q.fmt_messages(TEACHER_NAME))

        if q.never_got_feedback and use_openai:
            default = get_openai_feedback(q.original, q.messages[0].content, q.exo)
        elif q.never_got_feedback:
            default = "I dunno man, just work on it"
        else:
            default = ""

        # new_msg = st.chat_input("Feedback", key=q.uid)
        # if not new_msg and default:
            # st.session_state[q.uid] = default

        with st.form(key=f"form-{q.uid}"):
            new_msg = st.text_area("Feedback",
                                    value=default,
                                    height=250,
                                    key=q.uid)
            submit = st.form_submit_button("Send")
        if new_msg and submit:
            q.messages.append(Message(TEACHER_NAME, new_msg))
            st.rerun()

    # Check for new questions every second
    old_db = deepcopy(db())
    while True:
        if old_db != db():
            st.rerun()
        sleep(1)


def main():
    user = st.session_state.get("user")
    st.write(f"{user!r}")

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
    db().setdefault(user, [
        [Question(user, q, exo) for q in exo.variations]
        for exo in EXERCISES
    ])

    for exo, qs in zip(EXERCISES, db()[user]):
        st.write(exo.instructions)

        for i, q in enumerate(qs):
            st.markdown(f"## {i+1}. {q.original}")

            with st.container():
                st.write(q.fmt_messages(user))
                if len(q.messages) != 1 and (new := st.chat_input(key=f"chat-{q.uid}")):
                    q.messages.append(Message(user, new))
                    st.rerun()

            # If there was never any feedback, don't show the following questions
            if q.never_got_feedback:
                break
        else:
            continue
        break

    # Check for new messages every second
    past_msgs = deepcopy(db()[user])
    while True:
        if past_msgs != db()[user]:
            st.rerun()
        sleep(1)



if __name__ == "__main__":
    main()
