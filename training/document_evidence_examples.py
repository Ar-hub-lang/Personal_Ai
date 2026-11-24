"""Fictional equipment records with confusable dates, people and table entries."""

import random


EQUIPMENT = {
    "train": (
        ("hydraulic press", "brake lathe", "Inez Moreno", "Omar Hart", "welding bay"),
        ("seismic recorder", "survey tripod", "Tara Quill", "Peter Ames", "instrument room"),
        ("parcel scale", "barcode wand", "Finn Murphy", "Asha Malik", "mail room"),
        ("mixing drum", "cooling tank", "Devi Rao", "Ezra King", "engine room"),
        ("film scanner", "slide viewer", "Nila Roy", "Marek Nowak", "archive room"),
        ("kiln controller", "glaze mixer", "Anya Vale", "Ahmed Reed", "ceramics studio"),
        ("card reader", "thermal printer", "Mateo Paz", "Leah Dunn", "accounting room"),
        ("pneumatic drill", "riveting tool", "Sara Neal", "Imani Cole", "assembly floor"),
        ("wind tunnel", "flow meter", "Anil Seth", "Iris Wong", "research wing"),
        ("ticket terminal", "payment reader", "Lina Shah", "Hugo West", "entrance lobby"),
        ("water sampler", "pressure gauge", "Priya Bell", "Kaito Sato", "test chamber"),
        ("vacuum chamber", "electric hoist", "Nia Moss", "Leo Vega", "service dock"),
        ("tape dispenser", "pallet jack", "Lara Chen", "Adam Finch", "packing station"),
        ("circuit tester", "signal monitor", "Sam Lee", "Ellen Ross", "repair depot"),
        ("potting bench", "seed tray", "Mary Kent", "Avi Kaur", "greenhouse"),
        ("paper cutter", "stapling machine", "Mina Park", "Dev Kapoor", "print shop"),
    ),
    "validation": (
        ("radar antenna", "tracking receiver", "Rue Hall", "Jin Aoki", "tower room"),
        ("air compressor", "dust extractor", "Zoe Shaw", "Ian Carr", "paint booth"),
        ("pedal lathe", "chisel grinder", "Dina Ansari", "Mark Orr", "wood shop"),
        ("oil pump", "grease injector", "Flo Silva", "Uday Menon", "lift shaft"),
    ),
    "test": (
        ("acoustic meter", "fiber spool", "Nell Cruz", "Mohan Gill", "audio booth"),
        ("turbidity meter", "float switch", "Alma Russo", "Noor Ali", "sample room"),
        ("pressure cooker", "steam kettle", "Sia Nair", "Ian Bloom", "test kitchen"),
        ("label printer", "foam cutter", "Maya Sol", "Owen Cho", "shipping desk"),
    ),
}


def document_evidence_examples(split: str, abstain: int) -> list[tuple[str, list[str], int]]:
    rng = random.Random({"train": 81, "validation": 97, "test": 113}[split])
    examples = []
    for equipment, other, owner, colleague, place in EQUIPMENT[split]:
        commissioning_year = rng.randrange(2007, 2015)
        inspection_year = rng.randrange(2021, 2026)
        purchase_year = rng.randrange(2016, 2021)
        installation = f"The {equipment} entered service in {commissioning_year}."
        inspection = f"Inspectors audited the {equipment} in {inspection_year}."
        purchase = f"The team ordered the {other} in {purchase_year}."
        dismissal = f"The {other} was removed from active use."
        person = f"{owner} manages reservations for the {equipment}."
        reviewer = f"{colleague} processes fault reports for the {equipment}."
        location = f"Return borrowed {equipment} accessories to the cabinet in the {place}."
        cost = rng.randrange(14, 60)
        service_fee = f"Item | Service fee | Lead time\n{equipment.capitalize()} | ${cost} | Two days"
        other_fee = f"Item | Service fee | Lead time\n{other.capitalize()} | ${cost + 20} | One day"

        rows = (
            (f"When was the {equipment} commissioned?", installation, [inspection, purchase, dismissal]),
            (f"What year was the {equipment} inspected?", inspection, [installation, purchase, dismissal]),
            (f"What year was the {equipment} purchased?", None, [inspection, purchase, installation, dismissal]),
            (f"Who coordinates bookings for the {equipment}?", person, [reviewer, installation, dismissal]),
            (f"Who approved purchase of the {equipment}?", None, [person, reviewer, purchase, inspection]),
            (f"Where should accessories for the {equipment} go after use?", location, [reviewer, dismissal, inspection]),
            (f"How much does it cost to service the {equipment}?", service_fee, [other_fee, reviewer, purchase]),
            (f"What is the purchase price of the {equipment}?", None, [service_fee, other_fee, purchase, inspection]),
        )
        for question, answer, distractors in rows:
            passages = [answer, *distractors] if answer else distractors
            rng.shuffle(passages)
            examples.append((question, passages, passages.index(answer) if answer else abstain))
    return examples
