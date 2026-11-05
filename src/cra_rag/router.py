"""Route a question before any search (intent routing).

The assistant answers only from the regulation, so a question about the assistant
itself ("who are you?", "what can you do?") finds nothing there and gets a refusal.
Such questions are recognised here and get a fixed presentation, written in
advance: nothing to invent, no search, no model call. Everything else goes on to
the RAG pipeline or the agent.

The rules are deliberately narrow. A question must speak to the assistant ("your
role", "what can you do") AND contain no word of the regulation's domain, so that
"Quel est le rôle de l'ENISA ?" or "What can you do about a vulnerability?" still
go to the regulation. A test checks that none of the reference questions is
caught by mistake.
"""

import re
import unicodedata

ABOUT = {
    "fr": (
        "Je suis cra-rag, un assistant qui répond aux questions sur deux textes européens de "
        "cybersécurité des produits : le Cyber Resilience Act (règlement (UE) 2024/2847) et "
        "l'acte délégué RED (règlement délégué (UE) 2022/30).\n\n"
        "Mon rôle : retrouver dans ces textes les passages qui répondent à votre question, puis "
        "vous donner une réponse courte qui cite l'article ou l'annexe utilisé, par exemple "
        "[cra:art_14], pour que vous puissiez la vérifier.\n\n"
        "Mes limites : je ne connais que ces deux textes, en français et en anglais. Je ne "
        "couvre ni les autres lois (NIS2, RGPD...), ni les normes harmonisées EN 18031. Je peux "
        "me tromper, et je ne donne pas de conseil juridique.\n\n"
        "Exemple de question : « Dans quel délai faut-il signaler une vulnérabilité activement "
        "exploitée ? »"
    ),
    "en": (
        "I am cra-rag, an assistant that answers questions on two EU texts on product "
        "cybersecurity: the Cyber Resilience Act (Regulation (EU) 2024/2847) and the RED "
        "delegated act (Delegated Regulation (EU) 2022/30).\n\n"
        "My role: find the passages of these texts that answer your question, then give you a "
        "short answer that cites the article or annex used, for example [cra:art_14], so that "
        "you can check it.\n\n"
        "My limits: I only know these two texts, in English and French. I do not cover other "
        "laws (NIS2, GDPR...) nor the EN 18031 harmonised standards. I can be wrong, and I do "
        "not give legal advice.\n\n"
        'Example question: "What must a manufacturer report, and within what deadline, when '
        'an actively exploited vulnerability is found?"'
    ),
}

GREETING = re.compile(r"^(bonjour|bonsoir|salut|coucou|hello|hi|hey)\W*$")
ABOUT_THE_ASSISTANT = [
    re.compile(pattern)
    for pattern in (
        # French
        r"\bqui (es|etes)[ -]?(tu|vous)\b",
        r"\b(ton|ta|tes|votre|vos) (role|objectif|but|mission|capacites?|competences?"
        r"|limites?|fonction)\b",
        r"\b(que|qu'est-ce que) (sais|peux|savez|pouvez)[ -]?(tu|vous) faire\b",
        r"\bqu'est-ce que (tu|vous) (sais|peux|savez|pouvez) faire\b",
        r"\b(tu sers|vous servez) a quoi\b|\ba quoi (sers|servez)[ -]?(tu|vous)\b",
        r"\bpresente[ -]?(toi|vous)\b",
        r"\bcomment (tu fonctionnes|fonctionnes[ -]tu|vous fonctionnez|fonctionnez[ -]vous)\b",
        # English
        r"\bwho are you\b|\bwhat are you\b",
        r"\byour (role|goal|purpose|aim|job|capabilities|capability|skills|limits|limitations)\b",
        r"\bwhat (can|do) you do\b",
        r"\bintroduce yourself\b|\bhow do you work\b",
    )
]
# Words of the regulation's domain: a question holding one is about the regulation.
DOMAIN = re.compile(
    r"\b(cra|red|nis2|reglement|regulation|article|annexe?|fabricant|manufacturer|import"
    r"|distribut|produit|product|vulnerabilit|conformit|marquage|marking|certific|incident"
    r"|notifi|signal|report|enisa|csirt|norme|standard|amende|fine|sanction|logiciel"
    r"|software|mise a jour|update|securite|security|cybersecurit)",
)


def _plain(text: str) -> str:
    """Lower case, no accents, straight apostrophes, single spaces."""
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def is_about_assistant(question: str) -> bool:
    plain = _plain(question)
    if GREETING.match(plain):
        return True
    if DOMAIN.search(plain):
        return False
    return any(pattern.search(plain) for pattern in ABOUT_THE_ASSISTANT)
