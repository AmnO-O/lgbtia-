"""
Utility to generate synthetic sample data conforming to the official
StereoQueerEval SemEval 2027 schema for quick prototyping, smoke tests, and CI/CD.
"""

import os
import csv

SAMPLE_EN = [
    {
        "StereoQueerEval_id": "training_EN_0001",
        "yt_title": "Discussion on LGBTQ+ Rights and Community Healthcare",
        "yt_description": "A panel debate exploring current legislation and healthcare access for queer youth.",
        "yt_comment": "They are always trying to impose their agenda on everyone in the media.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_lgbtqia+"
    },
    {
        "StereoQueerEval_id": "training_EN_0002",
        "yt_title": "Discussion on LGBTQ+ Rights and Community Healthcare",
        "yt_description": "A panel debate exploring current legislation and healthcare access for queer youth.",
        "yt_comment": "Such an informative discussion, thank you for inviting diverse perspectives!",
        "stereotype": "no",
        "hate_speech": "no",
        "target": "none"
    },
    {
        "StereoQueerEval_id": "training_EN_0003",
        "yt_title": "Pride Month Parade Celebrations in City Center",
        "yt_description": "Highlights from the annual pride festivities and community gatherers.",
        "yt_comment": "All gay men are obsessed with drama and attention.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_g"
    },
    {
        "StereoQueerEval_id": "training_EN_0004",
        "yt_title": "Pride Month Parade Celebrations in City Center",
        "yt_description": "Highlights from the annual pride festivities and community gatherers.",
        "yt_comment": "These freaks should not be allowed anywhere near public streets, disgusting.",
        "stereotype": "no",
        "hate_speech": "yes_explicit",
        "target": "group_lgbtqia+"
    },
    {
        "StereoQueerEval_id": "training_EN_0005",
        "yt_title": "Interview with Non-Binary Author on New Memoir",
        "yt_description": "Exploring themes of identity, literature, and self-discovery.",
        "yt_comment": "Non-binary identity is just a made up trend for teenagers seeking attention.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_nb"
    },
    {
        "StereoQueerEval_id": "training_EN_0006",
        "yt_title": "Interview with Non-Binary Author on New Memoir",
        "yt_description": "Exploring themes of identity, literature, and self-discovery.",
        "yt_comment": "She is just confused and someone should stop her book from being published.",
        "stereotype": "no",
        "hate_speech": "yes_explicit",
        "target": "individual_nb"
    },
    {
        "StereoQueerEval_id": "training_EN_0007",
        "yt_title": "Documentary on Historic Transgender Activism",
        "yt_description": "Retracing the roots of 20th century equality movements.",
        "yt_comment": "Trans women are deceiving people and ruining real women sports.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_t"
    },
    {
        "StereoQueerEval_id": "training_EN_0008",
        "yt_title": "Documentary on Historic Transgender Activism",
        "yt_description": "Retracing the roots of 20th century equality movements.",
        "yt_comment": "Very moving documentary, learned a lot about history.",
        "stereotype": "no",
        "hate_speech": "no",
        "target": "none"
    }
]

SAMPLE_IT = [
    {
        "StereoQueerEval_id": "training_IT_0001",
        "yt_title": "Dibattito sui diritti civili e famiglie arcobaleno",
        "yt_description": "Tavola rotonda sulle nuove proposte di legge e tutela dei minori.",
        "yt_comment": "I gay pensano sempre e solo alle feste e alla moda, non sono capaci di serietà.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_g"
    },
    {
        "StereoQueerEval_id": "training_IT_0002",
        "yt_title": "Dibattito sui diritti civili e famiglie arcobaleno",
        "yt_description": "Tavola rotonda sulle nuove proposte di legge e tutela dei minori.",
        "yt_comment": "Bel servizio giornalistico, spiegato con molto rispetto ed equilibrio.",
        "stereotype": "no",
        "hate_speech": "no",
        "target": "none"
    },
    {
        "StereoQueerEval_id": "training_IT_0003",
        "yt_title": "Manifestazione per la parità di genere a Roma",
        "yt_description": "Migliaia di cittadini scendono in piazza per i diritti umani.",
        "yt_comment": "Questi pervertiti andrebbero tutti cacciati via dal nostro paese!",
        "stereotype": "no",
        "hate_speech": "yes_explicit",
        "target": "group_lgbtqia+"
    },
    {
        "StereoQueerEval_id": "training_IT_0004",
        "yt_title": "Manifestazione per la parità di genere a Roma",
        "yt_description": "Migliaia di cittadini scendono in piazza per i diritti umani.",
        "yt_comment": "Le lesbiche sono semplicemente donne che odiano gli uomini.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_l"
    }
]

SAMPLE_NL = [
    {
        "StereoQueerEval_id": "training_NL_0001",
        "yt_title": "Nieuwsuur: Toekomst van gelijke rechten in Nederland",
        "yt_description": "Reportage over inclusie en acceptatie op scholen en werkplekken.",
        "yt_comment": "Al die queer mensen overdrijven altijd en spelen steeds het slachtoffer.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_lgbtqia+"
    },
    {
        "StereoQueerEval_id": "training_NL_0002",
        "yt_title": "Nieuwsuur: Toekomst van gelijke rechten in Nederland",
        "yt_description": "Reportage over inclusie en acceptatie op scholen en werkplekken.",
        "yt_comment": "Goede reportage met duidelijke feiten en neutrale presentatie.",
        "stereotype": "no",
        "hate_speech": "no",
        "target": "none"
    },
    {
        "StereoQueerEval_id": "training_NL_0003",
        "yt_title": "Canal Pride Amsterdam hoogtepunten",
        "yt_description": "Feestelijke botenparade trekt honderdduizenden bezoekers.",
        "yt_comment": "Walgelijk volk, sluit ze allemaal op.",
        "stereotype": "no",
        "hate_speech": "yes_explicit",
        "target": "group_lgbtqia+"
    },
    {
        "StereoQueerEval_id": "training_NL_0004",
        "yt_title": "Canal Pride Amsterdam hoogtepunten",
        "yt_description": "Feestelijke botenparade trekt honderdduizenden bezoekers.",
        "yt_comment": "Biseksuelen kunnen gewoon nooit kiezen en zijn ontrouw.",
        "stereotype": "yes",
        "hate_speech": "yes_implicit",
        "target": "group_b"
    }
]

def generate_sample_dataset(data_dir: str = "data", multiplier: int = 15):
    """Generates synthetic multi-lingual sample TSV datasets with field quoting."""
    os.makedirs(data_dir, exist_ok=True)
    splits = [
        ("StereoQueerEval_EN_training.tsv", SAMPLE_EN),
        ("StereoQueerEval_IT_training.tsv", SAMPLE_IT),
        ("StereoQueerEval_NL_training.tsv", SAMPLE_NL),
    ]

    for fname, sample_list in splits:
        records = []
        for i in range(multiplier):
            for row in sample_list:
                item = dict(row)
                item["StereoQueerEval_id"] = f"{item['StereoQueerEval_id']}_{i+1:03d}"
                records.append(item)

        out_path = os.path.join(data_dir, fname)
        if records:
            fieldnames = list(records[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(records)
            print(f"Generated {len(records)} sample rows at {out_path}")

if __name__ == "__main__":
    generate_sample_dataset()
