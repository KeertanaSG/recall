"""A large synthetic history with known answers hidden among realistic noise.

Noise rows are generated from a fixed seed. Needles are hand-written facts, and their
questions deliberately avoid the words in the needle, as a real vague question would.
"""

import random
from dataclasses import dataclass

from recall.storage.store import Store

TABLES = {
    "movies": ("Movies watched", {"title": "text", "rating": "real", "genre": "text", "watched_with": "text", "notes": "text"}),
    "books": ("Books read", {"title": "text", "author": "text", "rating": "real", "notes": "text"}),
    "restaurants": ("Places eaten at", {"name": "text", "area": "text", "rating": "real", "notes": "text"}),
    "expenses": ("Money spent", {"item": "text", "amount": "real", "category": "text", "spent_on": "date", "notes": "text"}),
    "loans": ("Money lent or borrowed", {"person": "text", "amount": "real", "direction": "text", "reason": "text", "repaid": "boolean"}),
    "tasks": ("Tasks and promises", {"task": "text", "person": "text", "due_on": "date", "done": "boolean"}),
    "people": ("Facts about people", {"name": "text", "fact": "text"}),
    "trips": ("Trips taken", {"place": "text", "notes": "text"}),
    "notes": ("General notes", {"body": "text"}),
}

NAMES = ["Arjun", "Priya", "Sam", "Kavya", "Rahul", "Divya", "Nikhil", "Sneha", "Karan", "Pooja", "Aditya", "Isha"]
TITLES = ["The Long Road", "Midnight Harbour", "Paper Kites", "City of Echoes", "Silent River", "Golden Hour",
          "The Last Train", "Broken Compass", "Summer Rain", "Glass Houses", "Second Chances", "Wild Orchard"]
GENRES = ["drama", "comedy", "thriller", "action", "romance", "documentary", "horror", "animation"]
OPINIONS = ["pretty good", "bit slow", "fun watch", "forgettable", "solid ending", "overrated", "liked the music", "okay"]
AUTHORS = ["R. Menon", "L. Hart", "S. Kapoor", "J. Ellis", "M. Rao", "T. Novak"]
AREAS = ["Koramangala", "Jayanagar", "HSR", "Whitefield", "Malleshwaram", "Indiranagar"]
DISHES = ["biryani", "dosa", "burger", "pasta", "momos", "thali", "tacos", "ramen", "sandwich"]
PLACES = ["Cafe", "Kitchen", "House", "Bistro", "Corner", "Express", "Grill", "Diner"]
ITEMS = [("cab", "transport"), ("uber", "transport"), ("auto", "transport"), ("coffee", "food"), ("groceries", "food"),
         ("lunch", "food"), ("dinner", "food"), ("electricity bill", "bills"), ("phone recharge", "bills"),
         ("internet bill", "bills"), ("t-shirt", "shopping"), ("shoes", "shopping"), ("movie tickets", "fun")]
TASKS = ["call {n} back", "send {n} the slides", "pay the maid", "book a haircut", "reply to {n}'s email", "buy milk",
         "clean the fridge", "water the balcony pots", "share the photos with {n}", "pick up laundry"]
FACTS = ["works in marketing", "lives near MG Road", "likes cricket", "has two kids", "plays guitar", "is vegetarian",
         "studies design", "supports RCB", "runs marathons", "drinks only black tea"]
TOWNS = ["Goa", "Mysore", "Ooty", "Pondicherry", "Jaipur", "Kochi", "Udaipur", "Mumbai"]
NOTES = ["meeting moved to Thursday", "try the new gym near office", "remember to back up photos", "podcast about history",
         "good stretch routine from youtube", "buy a bigger water bottle", "printer needs new ink", "renew library card"]


@dataclass(frozen=True)
class Needle:
    key: str
    subject: str
    table: str
    values: dict
    question: str


NEEDLES = [
    Needle("coco", "movies", "movies", {"title": "Coco", "rating": 9, "notes": "bawled at the ending, watched with mum"}, "Which film made me cry with my mother?"),
    Needle("whiplash", "movies", "movies", {"title": "Whiplash", "rating": 8, "notes": "left the theatre with my heart racing"}, "What movie really stressed me out?"),
    Needle("paddington", "movies", "movies", {"title": "Paddington 2", "rating": 8, "notes": "watched on the flight to Delhi"}, "What did I watch on a plane?"),
    Needle("arrival", "movies", "movies", {"title": "Arrival", "rating": 10, "genre": "sci-fi", "notes": "aliens and linguistics"}, "Which film was about learning to talk to extraterrestrials?"),
    Needle("habits", "books", "books", {"title": "Atomic Habits", "author": "James Clear", "notes": "got me to start journaling daily"}, "Which book changed my morning routine?"),
    Needle("overstory", "books", "books", {"title": "The Overstory", "author": "Richard Powers", "notes": "Meera recommended it"}, "What was the novel my friend Meera suggested?"),
    Needle("sapiens", "books", "books", {"title": "Sapiens", "author": "Yuval Noah Harari", "notes": "human history, loved the farming chapter"}, "What did I read about how civilisation developed?"),
    Needle("karavalli", "food", "restaurants", {"name": "Karavalli", "rating": 9, "notes": "best fish curry, took parents for their anniversary"}, "Where did I take mom and dad for their special day?"),
    Needle("dyu", "food", "restaurants", {"name": "Dyu Art Cafe", "rating": 8, "notes": "quiet, good wifi, can sit with a laptop for hours"}, "Is there somewhere calm I can work from?"),
    Needle("nagarjuna", "food", "restaurants", {"name": "Spice Garden", "rating": 3, "notes": "way too hot, stomach upset the next day"}, "Which place made me sick?"),
    Needle("desk", "money", "expenses", {"item": "standing desk", "amount": 18500, "category": "home"}, "What did I buy for my home office?"),
    Needle("rohan", "money", "loans", {"person": "Rohan", "amount": 5000, "direction": "lent", "reason": "his rent, returns after bonus"}, "Who needed help with housing costs?"),
    Needle("vet", "money", "expenses", {"item": "vet visit for Bruno", "amount": 2400, "category": "pets"}, "How much was the dog's doctor?"),
    Needle("insurance", "tasks", "tasks", {"task": "renew car insurance before it lapses", "due_on": "2026-12-14"}, "Is anything coming up for my vehicle?"),
    Needle("resume", "tasks", "tasks", {"task": "review Neha's resume", "person": "Neha"}, "What did I say I'd do for the friend who's job hunting?"),
    Needle("passport_appt", "tasks", "tasks", {"task": "collect the visa documents from the embassy", "due_on": "2026-12-02"}, "Do I have any paperwork to pick up for travel?"),
    Needle("peanuts", "people", "people", {"name": "Meera", "fact": "allergic to peanuts"}, "Who can't eat nuts?"),
    Needle("vikram", "people", "people", {"name": "Vikram", "fact": "moved to Berlin, works at a bike startup"}, "Which friend lives in Germany?"),
    Needle("ananya", "people", "people", {"name": "Ananya", "fact": "birthday 12 March, loves pottery"}, "Who would like a ceramics gift?"),
    Needle("coorg", "trips", "trips", {"place": "Coorg", "notes": "homestay on a coffee plantation"}, "Where did we stay among the estates?"),
    Needle("hampi", "trips", "trips", {"place": "Hampi", "notes": "rented scooters, went bouldering at sunrise"}, "Which trip had rock climbing?"),
    Needle("key", "notes", "notes", {"body": "spare house key is with Mrs Rao next door"}, "Who has my extra key?"),
    Needle("plants", "notes", "notes", {"body": "idea: app that tracks when to water each plant"}, "What was my startup idea about gardening?"),
    Needle("filter", "notes", "notes", {"body": "AC filter last cleaned in October"}, "When did I last service the air conditioner?"),
]

# Nothing in the history answers these, in noise or in needles.
ABSENT = [
    "What is my passport number?",
    "What's my blood group?",
    "When is my sister's wedding?",
    "What did I do in Japan?",
    "How much did I spend on skiing gear?",
    "Who is my dentist?",
    "What's the opera I saw last year?",
    "Which of my friends owns a Tesla?",
    "What did I name my boat?",
    "Did I ever go to Iceland?",
]


def _noise(rng: random.Random, table: str) -> tuple[dict, str]:
    name, title = rng.choice(NAMES), f"{rng.choice(TITLES)} {rng.randint(1, 3)}"
    day = f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}"
    if table == "movies":
        return {"title": title, "rating": rng.randint(3, 10), "genre": rng.choice(GENRES),
                "watched_with": name, "notes": rng.choice(OPINIONS)}, f"watched {title}"
    if table == "books":
        return {"title": title, "author": rng.choice(AUTHORS), "rating": rng.randint(3, 10),
                "notes": rng.choice(OPINIONS)}, f"read {title}"
    if table == "restaurants":
        place = f"{rng.choice(TITLES).split()[-1]} {rng.choice(PLACES)}"
        return {"name": place, "area": rng.choice(AREAS), "rating": rng.randint(4, 10),
                "notes": f"good {rng.choice(DISHES)}"}, f"ate at {place}"
    if table == "expenses":
        item, category = rng.choice(ITEMS)
        return {"item": item, "amount": rng.randint(30, 3000), "category": category, "spent_on": day}, f"{item}"
    if table == "loans":
        return {"person": name, "amount": rng.randint(100, 3000), "direction": rng.choice(["lent", "borrowed"]),
                "reason": rng.choice(DISHES), "repaid": rng.randint(0, 1)}, f"loan with {name}"
    if table == "tasks":
        task = rng.choice(TASKS).format(n=name)
        return {"task": task, "person": name, "due_on": day, "done": 1}, task
    if table == "people":
        return {"name": name, "fact": rng.choice(FACTS)}, f"{name} {rng.choice(FACTS)}"
    if table == "trips":
        return {"place": rng.choice(TOWNS), "notes": rng.choice(OPINIONS)}, "trip"
    return {"body": rng.choice(NOTES)}, "note"


NOISE_COUNTS = {"movies": 800, "books": 300, "restaurants": 500, "expenses": 2500, "loans": 150,
                "tasks": 500, "people": 200, "trips": 100, "notes": 450}


def build_history(store: Store, seed: int = 7) -> dict[str, tuple[str, int]]:
    """Fill a store and return where each needle landed, as {key: (table, row_id)}."""
    store.sync_embeddings = lambda: None  # Embed everything once at the end instead.
    rng = random.Random(seed)
    for table, (description, columns) in TABLES.items():
        store.create_table(
            table,
            description,
            [{"name": name, "type": kind} for name, kind in columns.items()],
            confirmed_distinct=True,
        )
    rows = [(table, *_noise(rng, table)) for table, count in NOISE_COUNTS.items() for _ in range(count)]
    planted = [(needle.table, needle.values, needle.key) for needle in NEEDLES]
    positions = sorted(rng.sample(range(len(rows)), len(planted)))
    locations = {}
    for index, (table, values, source_text) in enumerate(rows):
        if index in positions:
            needle_table, needle_values, key = planted[positions.index(index)]
            locations[key] = (needle_table, store.add_row(needle_table, needle_values, ""))
        store.add_row(table, values, source_text)
        store.turn_id = None
    del store.sync_embeddings
    store.sync_embeddings()
    return locations
