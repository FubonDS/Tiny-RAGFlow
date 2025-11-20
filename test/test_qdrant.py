from src.core.qdrant_index import QdrantIndex
from src.core.client.embedding_rerank_client import MultiVectorModel
import numpy as np  
import asyncio
from fastembed import LateInteractionTextEmbedding

model = MultiVectorModel(model_path="colbert-ir/colbertv2.0")
descriptions = ["In 1431, Jeanne d'Arc is placed on trial on charges of heresy. The ecclesiastical jurists attempt to force Jeanne to recant her claims of holy visions.",
 "A film projectionist longs to be a detective, and puts his meagre skills to work when he is framed by a rival for stealing his girlfriend's father's pocketwatch.",
 "A group of high-end professional thieves start to feel the heat from the LAPD when they unknowingly leave a clue at their latest heist.",
 "A petty thief with an utter resemblance to a samurai warlord is hired as the lord's double. When the warlord later dies the thief is forced to take up arms in his place.",
 "A young boy named Kubo must locate a magical suit of armour worn by his late father in order to defeat a vengeful spirit from the past.",
 "A biopic detailing the 2 decades that Punjabi Sikh revolutionary Udham Singh spent planning the assassination of the man responsible for the Jallianwala Bagh massacre.",
 "When a machine that allows therapists to enter their patients' dreams is stolen, all hell breaks loose. Only a young female therapist, Paprika, can stop it.",
 "An ordinary word processor has the worst night of his life after he agrees to visit a girl in Soho whom he met that evening at a coffee shop.",
 "A story that revolves around drug abuse in the affluent north Indian State of Punjab and how the youth there have succumbed to it en-masse resulting in a socio-economic decline.",
 "A world-weary political journalist picks up the story of a woman's search for her son, who was taken away from her decades ago after she became pregnant and was forced to live in a convent.",
 "Concurrent theatrical ending of the TV series Neon Genesis Evangelion (1995).",
 "During World War II, a rebellious U.S. Army Major is assigned a dozen convicted murderers to train and lead them into a mass assassination mission of German officers.",
 "The toys are mistakenly delivered to a day-care center instead of the attic right before Andy leaves for college, and it's up to Woody to convince the other toys that they weren't abandoned and to return home.",
 "A soldier fighting aliens gets to relive the same day over and over again, the day restarting every time he dies.",
 "After two male musicians witness a mob hit, they flee the state in an all-female band disguised as women, but further complications set in.",
 "Exiled into the dangerous forest by her wicked stepmother, a princess is rescued by seven dwarf miners who make her part of their household.",
 "A renegade reporter trailing a young runaway heiress for a big story joins her on a bus heading from Florida to New York, and they end up stuck with each other when the bus leaves them behind at one of the stops.",
 "Story of 40-man Turkish task force who must defend a relay station.",
 "Spinal Tap, one of England's loudest bands, is chronicled by film director Marty DiBergi on what proves to be a fateful tour.",
 "Oskar, an overlooked and bullied boy, finds love and revenge through Eli, a beautiful but peculiar girl."]

descriptions_embeddings = asyncio.run(model.embed_documents(descriptions))

index = QdrantIndex(config_path="./config/qdrant.yaml", auto_load=False)

index.add_batch(
    descriptions_embeddings,
    [{"id": f"desc_{i}", "text": descriptions[i]} for i in range(len(descriptions))],
    batch_size=5
)

query = "A movie for kids with fantasy elements and wonders"
query_vec = asyncio.run(model.embed_query(query))
scores, docs = index.search(query_vec, top_k=3)
print("\n==== Search Result ====")
for s, d in zip(scores, docs):
    print(f"{s:.4f}  →  {d}")

query_list = [
    "A young boy named Kubo must locate a magical suit of armour worn by his late father in order to defeat a vengeful spirit from the past.",
    "A story that revolves around drug abuse in the affluent north Indian State of Punjab and how the youth there have succumbed to it en-masse resulting in a socio-economic decline."
]

query_vec = asyncio.run(model.embed_query_batch(query_list))

scores, docs = index.search_batch(query_vec, top_k=3)
print("\n==== Batch Search Result ====")
for i in range(len(query_vec)):
    print(f"Query {i+1}:")
    for s, d in zip(scores[i], docs[i]):
        print(f"{s:.4f}  →  {d}")
    print("-----")

# list(embedding_model.query_embed("A movie for kids with fantasy elements and wonders"))[0]