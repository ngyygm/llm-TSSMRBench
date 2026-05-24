"""Generate planted narrative benchmark.

Inserts state-change paragraphs into a novel's chunk stream as independent
retrieval units, then generates questions that test whether a system can
retrieve the correct temporal version among highly similar distractors.

Design:
  - Full novel text is chunked into 512-word segments
  - For each scenario, 2-3 state paragraphs are written describing the same
    subject at different temporal points (e.g., an object before/after change)
  - State paragraphs are inserted as independent chunks at well-separated
    positions throughout the novel
  - Questions reference unique details that only appear in one state
  - No question mentions position, chapter, or version identifiers

Output:
  - data/benchmark_narrative/novel_chunks.jsonl  (all chunks: original + planted)
  - data/benchmark_narrative/questions.jsonl     (questions with gold/distractor refs)
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Scenario definitions: 20 groups, each with 2-3 states
# Each state has a unique paragraph + a question targeting a unique detail
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    # --- Scenario 1: The Blackwood pocket watch (3 states) ---
    {
        "id": "blackwood_watch",
        "subject": "the Blackwood pocket watch",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The Blackwood pocket watch was a handsome piece of craftsmanship, "
                    "its silver case polished to a mirror finish and its back engraved "
                    "with the image of a fox in mid-leap. Mr. Blackwood had purchased it "
                    "from a jeweler in Bath some twenty years prior, and it had kept "
                    "exemplary time ever since, never deviating more than a few seconds "
                    "in any given month. He carried it daily in his waistcoat pocket, "
                    "attached to a fine silver chain, and consulted it with the quiet "
                    "pride of a man who valued precision."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The once-handsome Blackwood pocket watch now bore a crack across "
                    "its crystal face, and the silver case showed deep scratches as "
                    "though it had been dragged across stone. The fox engraving on the "
                    "back had been worn nearly smooth by some misadventure, its outline "
                    "barely visible. More troubling still, the watch had taken to losing "
                    "several minutes each day, its mechanism evidently disturbed by "
                    "whatever calamity had befallen its exterior. Mr. Blackwood no "
                    "longer carried it with his former pride."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The Blackwood pocket watch had been sent to a skilled artisan in "
                    "London who replaced the damaged silver casing entirely with one of "
                    "polished gold. While the familiar fox engraving had been restored "
                    "to the back, the craftsman had rendered it in a more angular, less "
                    "fluid style than the original, giving the creature a stiffer "
                    "appearance. The crystal was replaced and the mechanism recalibrated, "
                    "so that the watch once again kept time with remarkable precision, "
                    "though its owner sometimes frowned at the slight difference in the "
                    "engraving's character."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "Where was the Blackwood pocket watch originally purchased?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "By how much time does the damaged Blackwood pocket watch deviate each day?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "How does the restored engraving on the Blackwood watch differ in style from the original?",
                "difficulty": "high",
            },
        ],
    },
    # --- Scenario 2: The Ashworth ruby brooch (3 states) ---
    {
        "id": "ashworth_brooch",
        "subject": "the Ashworth ruby brooch",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The Ashworth ruby brooch was a piece of considerable distinction, "
                    "featuring a deep red stone of remarkable clarity set in a delicate "
                    "silver filigree. It had been passed down through four generations "
                    "of Ashworth women and was worn by the present Mrs. Ashworth at "
                    "every assembly and dinner party of note. The ruby caught the "
                    "candlelight with a particular warmth that drew compliments from "
                    "even the most indifferent observers, and its setting was so fine "
                    "that many a jeweler had offered to purchase it on sight."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "Following the unfortunate incident at the Michaelmas ball, the "
                    "Ashworth ruby brooch had been secretly replaced with a glass "
                    "imitation. The original ruby had come loose from its setting during "
                    "a vigorous reel and was lost somewhere on the dance floor, never to "
                    "be recovered despite a thorough search the following morning. The "
                    "replacement paste stone was of tolerable quality—only the family "
                    "knew the truth—and Mrs. Ashworth wore it with the same confidence "
                    "as before, though she avoided allowing anyone to examine it closely."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "By a remarkable stroke of fortune, the original Ashworth ruby was "
                    "discovered some months later in a pawnbroker's shop in the neighboring "
                    "town, where it had been sold by a person unknown. The stone was "
                    "retrieved and reset—not in its original silver filigree, but in a "
                    "new gold setting chosen by Mrs. Ashworth herself. Close inspection "
                    "revealed a tiny chip on one facet of the ruby, a memento of its "
                    "adventure, but its color and fire were undiminished."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "How many generations had the Ashworth ruby brooch been passed down through?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What event caused the Ashworth ruby to be lost from its brooch?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What mark did the recovered Ashworth ruby bear from its time away?",
                "difficulty": "high",
            },
        ],
    },
    # --- Scenario 3: The Millbrook village bridge (2 states) ---
    {
        "id": "millbrook_bridge",
        "subject": "the Millbrook village bridge",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The bridge at Millbrook had stood for upwards of a century, a "
                    "sturdy structure of grey stone spanning the river in three graceful "
                    "arches. At each corner stood a carved stone gargoyle, their faces "
                    "worn smooth by decades of weather but still bearing a hint of their "
                    "original fierce expressions. The bridge was wide enough for two "
                    "carriages to pass abreast, and its parapets were a favored spot for "
                    "the village children to lean and watch the water flow beneath."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The great flood of November had taken the Millbrook bridge entirely, "
                    "tearing away two of its three stone arches and sweeping the venerable "
                    "gargoyles downstream. In the weeks that followed, a temporary wooden "
                    "structure was erected by the parish—a humble thing of rough-hewn "
                    "planks and roped railings that swayed uneasily under the weight of a "
                    "horse. The villagers crossed it with caution, speaking wistfully of "
                    "the solid stone that had connected their lives for so long."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What decorative carvings stood at the corners of the Millbrook bridge?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What was the temporary Millbrook crossing made of after the flood?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 4: Mrs. Crawford's rose garden (3 states) ---
    {
        "id": "crawford_roses",
        "subject": "Mrs. Crawford's rose garden",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "Mrs. Crawford's rose garden was the undisputed pride of the parish, "
                    "containing no fewer than thirty distinct varieties arranged in "
                    "meticulous rows according to color and season. Her white roses were "
                    "particularly celebrated, winning prizes at the county fair three "
                    "years running. Neighbors traveled from adjoining villages merely to "
                    "walk along the gravel paths and admire the blooms, and Mrs. Crawford "
                    "maintained a leather-bound journal in which she recorded the "
                    "progress of each variety with the dedication of a naturalist."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The blight that struck in June was merciless. By July, only five "
                    "varieties of Mrs. Crawford's once-magnificent roses survived—the "
                    "hardiest specimens, which even the disease could not wholly conquer. "
                    "The gravel paths were overgrown with weeds, and the leather journal "
                    "lay closed on her parlor table, its record of happier seasons too "
                    "painful to consult. The white roses that had won such acclaim were "
                    "among the first to succumb, their petals browning even on the stem."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "By the following spring, Mrs. Crawford's garden had been entirely "
                    "replanted with new varieties donated by sympathetic neighbors from "
                    "six surrounding parishes. Not a single one of the original thirty "
                    "varieties remained—the blight had seen to that—but the new "
                    "collection was beginning to flourish in its own right. Mrs. Crawford "
                    "had begun a fresh journal, its first entry acknowledging the kindness "
                    "of the community that had helped restore what nature had taken away."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "How many varieties of roses did Mrs. Crawford originally cultivate in her garden?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "Which of Mrs. Crawford's prize roses were the first to succumb to the blight?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "How many of Mrs. Crawford's original rose varieties survived to the replanting?",
                "difficulty": "high",
            },
        ],
    },
    # --- Scenario 5: The Foxworth estate gates (2 states) ---
    {
        "id": "foxworth_gates",
        "subject": "the Foxworth estate gates",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The gates of Foxworth Hall were a landmark in themselves, standing "
                    "twelve feet high in wrought iron with the family motto 'Fortune "
                    "Favours the Bold' worked into the iron scrollwork above the central "
                    "arch. They had been forged in the previous century by a renowned "
                    "blacksmith from York, and their paint—a deep green—was touched up "
                    "each spring by the estate's groundskeeper. Carriages approaching "
                    "from the lane could see the gates from a quarter mile away, their "
                    "silhouette unmistakable against the sky."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The iron gates of Foxworth Hall had been removed in their entirety "
                    "and contributed to the war effort, leaving only the stone gateposts "
                    "standing bare and forlorn at the entrance to the drive. In their "
                    "place, the groundskeeper had fashioned a pair of wooden gates from "
                    "oak planks, sturdy enough but entirely plain, without inscription or "
                    "ornament. The motto that had once announced the family's creed was "
                    "nowhere in evidence, and the new gates bore only the marks of the "
                    "saw and hammer that had made them."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What motto was inscribed on the iron gates of Foxworth Hall?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What became of the iron from the original Foxworth estate gates?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 6: Captain Aldridge's sea chest (3 states) ---
    {
        "id": "aldridge_chest",
        "subject": "Captain Aldridge's sea chest",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "Captain Aldridge's sea chest occupied a place of reverence in the "
                    "corner of his study, a handsome piece of dark mahogany bound with "
                    "brass bands and secured by a lock of curious design that defied "
                    "every key in the house. It had accompanied him on his voyages to the "
                    "East Indies and the Mediterranean, and its contents—whatever they "
                    "might be—were a matter of endless speculation among the household. "
                    "The Captain never spoke of it and never opened it in the presence of "
                    "another soul."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "Upon Captain Aldridge's passing, the sea chest was opened at last "
                    "by Mr. Thornhill, the family solicitor, in the presence of two "
                    "witnesses. Inside were a collection of nautical charts charting "
                    "coastlines from the Cape of Good Hope to the Strait of Malacca, "
                    "and beneath these a sealed letter addressed to a name no one "
                    "present recognized. The brass-bound chest, so long a mystery, "
                    "yielded its secrets with surprising mundanity."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "After the solicitor had catalogued the charts and taken possession "
                    "of the sealed letter, the sea chest was found to contain only two "
                    "remaining items: a single dried flower, pressed flat and brittle "
                    "with age—some tropical species unknown to anyone in the "
                    "household—and a small ambrotype photograph of a young woman in "
                    "foreign dress, her image fading to silver. These mute artifacts "
                    "spoke of a life the Captain had never shared with his family."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What material were the bands on Captain Aldridge's sea chest?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "Who opened Captain Aldridge's sea chest after his death?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What photographic format was found among the remaining items in Captain Aldridge's chest?",
                "difficulty": "high",
            },
        ],
    },
    # --- Scenario 7: The Thornwood stained glass window (3 states) ---
    {
        "id": "thornwood_window",
        "subject": "the Thornwood church stained glass window",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The east window of Thornwood church was a celebrated example of "
                    "stained glass depicting the martyrdom of Saint George, with the "
                    "saint mounted on a white charger, his lance pointed at a green "
                    "dragon coiled beneath the hooves of his rearing horse. The reds and "
                    "blues of the composition were particularly vivid, casting colored "
                    "light across the altar throughout the morning hours. The window had "
                    "been donated by the Thornwood family in the previous century and was "
                    "regarded as the finest artwork in the county."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The great storm of autumn had sent a branch crashing through the "
                    "east window of Thornwood church, shattering the lower portion of "
                    "the stained glass completely. The dragon's tail was entirely lost, "
                    "and Saint George's lance had been broken into fragments too small to "
                    "salvage. Pieces of colored glass lay scattered across the altar and "
                    "the floor of the chancel, their reds and blues now mere shards "
                    "glinting in the rain that fell through the gap."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The restored east window of Thornwood church bore little resemblance "
                    "to its celebrated predecessor. The glazier, unable to match the "
                    "original design from the surviving fragments, had replaced the dragon "
                    "with a lamb resting at the saint's feet—an emblem of peace rather "
                    "than conquest. The parishioners received the change with mixed "
                    "feelings: some found the new scene more fitting for a house of "
                    "worship, while others mourned the loss of the fierce dragon that had "
                    "captivated their childhood imaginations."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What color was the dragon in the original Thornwood church stained glass window?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What part of the Thornwood church window was completely destroyed by the storm?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What animal replaced the dragon in the restored Thornwood church window?",
                "difficulty": "low",
            },
        ],
    },
    # --- Scenario 8: Dr. Pemberton's medical journal (2 states) ---
    {
        "id": "pemberton_journal",
        "subject": "Dr. Pemberton's medical journal",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "Dr. Pemberton's medical journal was a model of professional "
                    "diligence—a thick leather-bound volume in which he recorded, in "
                    "precise copperplate handwriting, the symptoms, treatments, and "
                    "outcomes of every patient he attended. His notes on the progress of "
                    "country fevers were detailed enough to serve as a reference for any "
                    "physician, and his sketches of anatomical observations were rendered "
                    "with a draftsman's care. He kept the journal locked in his surgery "
                    "cabinet alongside his most valued instruments."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The flood that inundated Dr. Pemberton's surgery had rendered his "
                    "medical journal entirely illegible—its pages swollen and stuck "
                    "together, the careful copperplate writing reduced to running ink "
                    "stains. He had since begun a new volume, hastily transcribing what "
                    "he could recall of his most important cases from memory. The new "
                    "journal, bound in plain cloth rather than leather, contained only "
                    "approximations of the detailed observations that had been lost, and "
                    "the anatomical sketches were perfunctory at best."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What handwriting style did Dr. Pemberton use in his original medical journal?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What caused the destruction of Dr. Pemberton's medical records?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 9: The Kingscross market bell (3 states) ---
    {
        "id": "kingscross_bell",
        "subject": "the Kingscross market bell",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The Kingscross market bell had called traders to their stalls for "
                    "generations, its clear bright tone audible from one end of the "
                    "market square to the other. It was a substantial piece of bronze "
                    "cast in a single pour, engraved with the year of its making and the "
                    "name of the parish. The bell-ringer, old Mr. Hodge, took great pride "
                    "in the precision of his daily ring, and the sound had become so "
                    "associated with the rhythm of market life that the villagers set "
                    "their watches by it."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The Kingscross market bell had developed a crack that ran in a "
                    "jagged line from rim to crown, and its once-clear tone was now a "
                    "flat, discordant sound that set the dogs to howling. Mr. Hodge had "
                    "retired from his bell-ringing duties in disgust, declaring that no "
                    "self-respecting bell should make such a noise. The crack had appeared "
                    "after a particularly hard frost, and the parish was divided on "
                    "whether to repair it or replace it entirely."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The cracked Kingscross market bell had been melted down and recast "
                    "by a foundry in the county town, emerging from the process as a "
                    "smaller but clearer-toned handbell that was presented to the market "
                    "inspector for his personal use. The new handbell bore a fresh "
                    "inscription commemorating the recasting, and its bright chime—though "
                    "lacking the carrying power of the original—was agreeably musical "
                    "and free of any hint of the old bell's cracked dissonance."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "Who was responsible for ringing the Kingscross market bell each day?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What weather event caused the crack in the Kingscross market bell?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What was the Kingscross market bell repurposed into after being recast?",
                "difficulty": "low",
            },
        ],
    },
    # --- Scenario 10: The Pennington tapestry (3 states) ---
    {
        "id": "pennington_tapestry",
        "subject": "the Pennington tapestry",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The Pennington tapestry was a magnificent wall hanging of "
                    "considerable age, depicting a medieval hunt in vivid detail: "
                    "horsemen in green tunics galloping across a stitched landscape, "
                    "their hounds in full cry pursuing a stag through a forest of "
                    "deep green and gold thread. The workmanship was extraordinary, "
                    "each horse individually rendered with flowing manes and the riders' "
                    "faces bearing distinct expressions of exhilaration. It dominated the "
                    "east wall of the Pennington drawing room and was the first thing "
                    "visitors remarked upon."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "Removed from the drawing room wall after the death of the elder "
                    "Mrs. Pennington, the tapestry was discovered to be badly damaged "
                    "by moths—whole sections of the hunt scene had been eaten away, "
                    "the horses reduced to ghostly outlines and the hounds vanished "
                    "entirely. It had been rolled up and stored in a damp attic for "
                    "several years, during which time the damage had worsened "
                    "considerably, and mildew had stained the once-bright gold threads "
                    "a murky brown."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The restored tapestry now hung once more in the Pennington drawing "
                    "room, though it bore little resemblance to the hunting scene that "
                    "had once captivated visitors. The restorer, working with what "
                    "fragments remained serviceable, had replaced the hunt with a pastoral "
                    "landscape—sheep grazing in a meadow beneath a peaceful sky, "
                    "the threads fresh and bright against the surviving sections of the "
                    "original background. The result was undeniably attractive, if no "
                    "longer the same work of art."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What animal were the hounds pursuing in the original Pennington tapestry?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "Where was the damaged Pennington tapestry stored after being removed from the wall?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What scene did the restorer create to replace the hunt in the Pennington tapestry?",
                "difficulty": "low",
            },
        ],
    },
    # --- Scenario 11: Lord Ashbury's stallion Thunder (3 states) ---
    {
        "id": "ashbury_thunder",
        "subject": "Lord Ashbury's stallion Thunder",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "Thunder was Lord Ashbury's most prized possession—a magnificent "
                    "black stallion of sixteen hands, with a white star on his forehead "
                    "and a temperament that was fiery on the racetrack but docile in the "
                    "stable. He was undefeated in the local steeplechases and had carried "
                    "his lordship to victory at the county meet three years in succession. "
                    "Grooms from neighboring estates came to admire his conformation, and "
                    "Lord Ashbury had refused several handsome offers for the animal, "
                    "declaring he would sooner part with his estate."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The fall at the November steeplechase had been a bad one. Thunder "
                    "had clipped the top of a fence and gone down heavily, throwing Lord "
                    "Ashbury clear but injuring his own left foreleg. The veterinarian "
                    "had tended to him with care, but the leg had not healed cleanly, and "
                    "the stallion now walked with a perceptible limp. His racing days were "
                    "finished, and Lord Ashbury visited him each morning in his stall, "
                    "grooming the dark coat himself and offering apples from his own hand."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "Thunder had adapted well to his retirement, living out his days in "
                    "a spacious paddock overlooking the Ashbury parkland. Though he could "
                    "no longer race, he had proven an exemplary breeding sire, siring "
                    "four foals in his first season—all inheriting their sire's dark coat "
                    "and three bearing the distinctive white star on their foreheads. Lord "
                    "Ashbury took particular delight in watching the foals at play, "
                    "seeing in their youthful galloping an echo of their father's former "
                    "glory."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "How many consecutive years did Lord Ashbury's stallion Thunder win the county steeplechase?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "Which of Thunder's legs was injured in the steeplechase fall?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "How many of Thunder's foals inherited his distinctive white forehead marking?",
                "difficulty": "high",
            },
        ],
    },
    # --- Scenario 12: The Fairweather music box (2 states) ---
    {
        "id": "fairweather_musicbox",
        "subject": "the Fairweather music box",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The Fairweather music box was a cherished heirloom, its walnut "
                    "case inlaid with mother-of-pearl flowers and its mechanism playing "
                    "a Mozart minuet when the lid was raised. Inside the lid, a "
                    "hand-painted pastoral scene depicted shepherdesses and their flock "
                    "beside a stream, the colors still fresh despite the box's age. "
                    "Miss Fairweather wound it each evening and let the melody play as "
                    "she read by the fire, finding in its delicate tones a comfort that "
                    "no other instrument could provide."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The Fairweather music box had been silent for many months, its "
                    "mechanism halted by a broken mainspring that the village clockmaker "
                    "lacked the skill to repair. Worse still, the hand-painted scene "
                    "inside the lid had faded to near-invisibility after years of "
                    "exposure to sunlight from the window where the box had rested. "
                    "The shepherdesses were now mere ghosts of color, and the walnut "
                    "case itself had developed a crack along one side. Miss Fairweather "
                    "could not bring herself to open the lid and confront its silence."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What composer's melody did the Fairweather music box play?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What mechanical failure caused the Fairweather music box to stop playing?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 13: The Thornwood village well (3 states) ---
    {
        "id": "thornwood_well",
        "subject": "the Thornwood village well",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The well at the center of Thornwood village was ancient, its stone "
                    "walls worn smooth by generations of rope and bucket. A wooden bucket "
                    "suspended by an iron winch drew water of exceptional clarity—so "
                    "clear that one could see the stones at the bottom, some twenty feet "
                    "below. The well was the gathering place for the village women each "
                    "morning, and its water was considered the finest in the district, "
                    "producing tea of a superior delicacy that visitors often remarked upon."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The Thornwood well had been sealed with a heavy stone slab after the "
                    "wooden bucket rotted through and fell into the depths, taking with it "
                    "several feet of ancient rope. The iron winch had rusted solid in its "
                    "housing, and no one in the village had the means or inclination to "
                    "restore it. Water was now drawn from the stream at the edge of the "
                    "village—a serviceable but inferior source that left a mineral taste "
                    "in the tea that old residents found disagreeable."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The Thornwood well had been permanently sealed with a cement cap on "
                    "the instructions of the parish council, and a small bronze plaque "
                    "affixed to the cap commemorated a child who had nearly fallen into "
                    "the open well the previous summer. The incident had galvanized the "
                    "village into action at last, and the stream remained the sole water "
                    "source. Flowers had been planted around the capped well, softening "
                    "its transformation from communal gathering place to memorial."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "How deep was the Thornwood village well?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What caused the Thornwood well's bucket to be lost?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What event prompted the permanent sealing of the Thornwood village well?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 14: Miss Templewater's watercolor painting (3 states) ---
    {
        "id": "templewater_painting",
        "subject": "Miss Templewater's watercolor painting",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "Miss Templewater's watercolor of the lake at dawn was her finest "
                    "achievement—a delicate composition showing swans gliding across "
                    "still water beneath a sky washed in pink and gold. She had spent "
                    "three mornings in succession capturing the effect of light on the "
                    "water, and the result had been entered in the county exhibition, "
                    "where it received an honorable mention. Her mother declared it too "
                    "good to sell and hung it in the best parlor where it caught the "
                    "afternoon light to particular advantage."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The watercolor that now hung in the Templewater parlor bore no "
                    "resemblance to the lake scene that had received the honorable "
                    "mention. Miss Templewater, in a fit of economy—or perhaps "
                    "dissatisfaction with her earlier work—had painted over the entire "
                    "canvas with a portrait of her younger sister Clara in a blue bonnet. "
                    "The swans and the dawn sky lay buried beneath layers of new pigment, "
                    "and only the original frame remained as evidence of the painting "
                    "that had once been."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The only remnant of Miss Templewater's original watercolor was a "
                    "small miniature that she had cut from the canvas before painting "
                    "it over—a square of perhaps three inches showing a single swan "
                    "in exquisite detail, all that remained of the dawn lake scene. "
                    "This fragment she had set into a gold locket that she wore on a "
                    "chain around her neck, a private memento of a work she had "
                    "otherwise chosen to destroy."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What birds appeared in Miss Templewater's original watercolor painting of the lake?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "Who was depicted in the portrait that Miss Templewater painted over her watercolor?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What piece of jewelry was made from the surviving fragment of Miss Templewater's painting?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 15: The Millbrook rectory wall (2 states) ---
    {
        "id": "millbrook_wall",
        "subject": "the Millbrook rectory wall",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The wall enclosing the Millbrook rectory garden was a modest "
                    "affair of local stone, standing no more than four feet high and "
                    "entirely covered in climbing roses that bloomed in extravagant "
                    "profusion each June. A wrought iron gate painted black provided "
                    "entrance from the lane, and the effect was so picturesque that "
                    "traveling artists frequently stopped to sketch it. The rector "
                    "attributed the roses' vigor to the favorable aspect and good soil, "
                    "though his wife insisted it was her careful pruning."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The boundary dispute with the neighboring landowner had resulted "
                    "in the rectory wall being torn down and rebuilt at twice its "
                    "original height in red brick—a material wholly foreign to the "
                    "Millbrook aesthetic. The climbing roses, their roots disturbed by "
                    "the demolition, had not survived the reconstruction, and the new "
                    "wall presented a bare and somewhat forbidding face to the lane. "
                    "The wrought iron gate was gone, replaced by a heavy wooden door "
                    "that suited the wall's new defensive character."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What flowers covered the original Millbrook rectory garden wall?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What caused the Millbrook rectory wall to be rebuilt?",
                "difficulty": "medium",
            },
        ],
    },
    # --- Scenario 16: The traveling troupe's stage wagon (3 states) ---
    {
        "id": "troupe_wagon",
        "subject": "the traveling troupe's stage wagon",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The stage wagon belonging to Harper's Travelling Players was "
                    "impossible to miss on the road: painted a vivid red and gold, "
                    "with the masks of Comedy and Tragedy rendered in bold strokes "
                    "upon its broad sides. The wagon carried costumes, props, and a "
                    "collapsible stage that could be erected in a market square within "
                    "the hour. Children ran alongside it as it passed through villages, "
                    "and the very sight of it was sufficient advertisement for the "
                    "evening's entertainment."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The Harper troupe's wagon had suffered a broken axle on the road "
                    "between villages and now stood forlorn at the roadside, its red "
                    "paint peeling and its theatrical masks faded by a summer's "
                    "exposure to sun and rain. The players had been forced to continue "
                    "on foot with what costumes they could carry, leaving the wagon "
                    "temporarily in the care of a sympathetic farmer. Its wheels were "
                    "sunk into the mud and its canvas cover torn, presenting a picture "
                    "of theatrical fortune reversed."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The Harper troupe's wagon had returned to the road at last, "
                    "repaired and repainted in a sober dark green that attracted no "
                    "particular attention. The theatrical masks of Comedy and Tragedy "
                    "that had once adorned its sides had been painted over entirely, "
                    "and the wagon now resembled any tradesman's cart. The players "
                    "confessed that the new appearance was practical but mourned the "
                    "loss of the flamboyant vehicle that had been their trademark."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What theatrical symbols were painted on the sides of the Harper troupe's original wagon?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What mechanical failure left the Harper troupe's wagon stranded on the road?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What color was the Harper troupe's wagon repainted after its repair?",
                "difficulty": "low",
            },
        ],
    },
    # --- Scenario 17: The Ravenscroft silver tea service (3 states) ---
    {
        "id": "ravenscroft_tea",
        "subject": "the Ravenscroft silver tea service",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The Ravenscroft tea service was a magnificent set of twelve "
                    "pieces in polished silver, each bearing the family crest—a "
                    "falcon grasping a quill. The set comprised teapot, coffeepot, "
                    "sugar bowl, creamer, waste bowl, two-handled tray, and six "
                    "matching cups with their saucers. It was produced only on the "
                    "most formal occasions, and the butler polished each piece "
                    "himself with a devotion that bordered upon reverence."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The Ravenscroft tea service was now incomplete, the sugar bowl "
                    "and creamer having been lost during the family's removal from "
                    "their townhouse to the country estate. Whether they had been "
                    "mislaid by the movers or stolen in transit could not be "
                    "determined, and the remaining ten pieces sat in their velvet-"
                    "lined case with two conspicuous gaps. The butler still polished "
                    "the surviving pieces with care, but his devotion was tinged with "
                    "sorrow at the incompleteness of the set."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The gaps in the Ravenscroft tea service had been filled at last "
                    "with replacement pieces purchased from a London silversmith—"
                    "matching in size and general appearance but bearing no family "
                    "crest. The new sugar bowl and creamer were fine enough in their "
                    "way, but the absence of the falcon-and-quill crest marked them "
                    "immediately as interlopers to anyone who looked closely. The "
                    "butler received them with a curt nod and placed them in the "
                    "velvet-lined case beside their elder companions."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What crest was engraved on the complete Ravenscroft tea service?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "Which pieces of the Ravenscroft tea service were lost during the move?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "How did the replacement pieces in the Ravenscroft tea service differ from the originals?",
                "difficulty": "low",
            },
        ],
    },
    # --- Scenario 18: The ancient oak at Foxworth field (2 states) ---
    {
        "id": "foxworth_oak",
        "subject": "the ancient oak at Foxworth field",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The great oak at the edge of Foxworth field was estimated by the "
                    "most knowledgeable woodsmen to be upwards of three centuries old. "
                    "Its trunk was so broad that three men linking hands could not have "
                    "encircled it, and its canopy spread over fully half the field, "
                    "providing shade for the cattle in summer and a roost for a "
                    "multitude of rooks. The tree was a landmark known throughout the "
                    "parish, and travelers used it to mark the turn toward the village."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The lightning strike had split the ancient oak from crown to root, "
                    "leaving half the tree standing in dead ruin while the other half "
                    "had fallen across the field boundary. The canopy that had once "
                    "shaded half the field was reduced to a ragged silhouette against "
                    "the sky. Yet from the surviving trunk, pale green shoots had begun "
                    "to emerge—new growth pushing through the charred bark with a "
                    "tenacity that the villagers regarded as miraculous."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "How old was the ancient oak at Foxworth field estimated to be?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What natural force destroyed the Foxworth field oak?",
                "difficulty": "low",
            },
        ],
    },
    # --- Scenario 19: Mr. Whitmore's traveling trunk (3 states) ---
    {
        "id": "whitmore_trunk",
        "subject": "Mr. Whitmore's traveling trunk",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "Mr. Whitmore's traveling trunk had returned from his Grand Tour "
                    "packed with botanical specimens collected across the continent: "
                    "pressed flowers from the Alps, seed pods from the Mediterranean, "
                    "and a small collection of dried herbs from a monastery garden in "
                    "northern Italy. Each specimen was carefully labeled in his neat "
                    "handwriting and interleaved with tissue paper. The trunk itself "
                    "was a sturdy leather-clad box bearing the stamps and labels of "
                    "hotels and railway stations from Rome to Geneva."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "Mr. Whitmore had emptied the trunk of its botanical contents and "
                    "repacked it entirely with woolen clothing—thick coats, scarves, "
                    "and sturdy boots—preparations for a walking expedition in the "
                    "Scottish Highlands that he had long contemplated. The hotel labels "
                    "still adorned the trunk's exterior, but within, the careful "
                    "specimens had been transferred to shelves in his study, replaced "
                    "by garments of a practical character entirely foreign to the "
                    "trunk's Continental past."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The traveling trunk now contained only Mr. Whitmore's personal "
                    "journals—seven volumes of observations and reflections accumulated "
                    "over his years of travel—and a single pressed edelweiss flower "
                    "preserved between the pages of the final journal. The woolen "
                    "clothing had been unpacked and stored, and the trunk sat in a "
                    "corner of his bedroom, its traveling days perhaps concluded, "
                    "serving as a quiet repository for the written record of his adventures."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What type of items from Italy were among Mr. Whitmore's botanical specimens?",
                "difficulty": "medium",
            },
            {
                "state_id": "b",
                "query_text": "What destination was Mr. Whitmore preparing for when he repacked his trunk with woolens?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "How many journals did Mr. Whitmore store in his traveling trunk?",
                "difficulty": "high",
            },
        ],
    },
    # --- Scenario 20: The hermitage in Ashbury Park (3 states) ---
    {
        "id": "ashbury_hermitage",
        "subject": "the hermitage in Ashbury Park",
        "states": [
            {
                "id": "a",
                "paragraph": (
                    "The hermitage in Ashbury Park was an ornamental folly of the kind "
                    "fashionable in the previous century—a picturesque thatched cottage "
                    "intended to suggest a hermit's rustic dwelling. It was furnished "
                    "with a wooden bench, a plain table, and several candlesticks, and "
                    "stood in a secluded grove of beech trees at the far end of the "
                    "pleasure grounds. Visitors to the park discovered it by following "
                    "a winding path and found its rustic charm a pleasing contrast to "
                    "the formality of the great house."
                ),
            },
            {
                "id": "b",
                "paragraph": (
                    "The hermitage in Ashbury Park had fallen into a state of advanced "
                    "neglect. Its thatched roof had partially collapsed, the wooden "
                    "bench was broken and covered in moss, and the candlesticks had "
                    "long since disappeared. The secluded grove that had once been its "
                    "charm now concealed its decay, and wandering dogs and foxes had "
                    "taken shelter within its crumbling walls. The winding path leading "
                    "to it was overgrown and scarcely passable."
                ),
            },
            {
                "id": "c",
                "paragraph": (
                    "The former hermitage in Ashbury Park had been converted to "
                    "practical use as a gardener's tool shed, its crumbling walls "
                    "repaired and a new slate roof replacing the collapsed thatch. The "
                    "interior now held iron shelving loaded with clay pots, trowels, "
                    "and twine, and the rustic charm of the ornamental folly had given "
                    "way to agricultural utility. The winding path had been cleared and "
                    "widened to accommodate a wheelbarrow."
                ),
            },
        ],
        "questions": [
            {
                "state_id": "a",
                "query_text": "What type of roof did the ornamental hermitage in Ashbury Park have?",
                "difficulty": "low",
            },
            {
                "state_id": "b",
                "query_text": "What kind of animals had taken shelter in the abandoned Ashbury Park hermitage?",
                "difficulty": "medium",
            },
            {
                "state_id": "c",
                "query_text": "What type of shelving was installed when the Ashbury Park hermitage was converted?",
                "difficulty": "high",
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Generic questions: temporal framing, no unique keywords
# These test whether retrieval can distinguish versions by semantic
# temporal cues rather than keyword matching.
# ---------------------------------------------------------------------------

GENERIC_QUESTIONS: dict[str, list[dict]] = {
    "blackwood_watch": [
        {"state_id": "a", "query_text": "What was the Blackwood pocket watch like in its original condition?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "How did the Blackwood pocket watch appear after it was damaged?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What changes were made when the Blackwood pocket watch was restored?", "difficulty": "high"},
    ],
    "ashworth_brooch": [
        {"state_id": "a", "query_text": "Describe the Ashworth ruby brooch as it originally existed.", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What became of the Ashworth ruby brooch after the incident at the ball?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "In what condition was the Ashworth ruby brooch returned after being recovered?", "difficulty": "high"},
    ],
    "millbrook_bridge": [
        {"state_id": "a", "query_text": "What was the bridge at Millbrook like originally?", "difficulty": "low"},
        {"state_id": "b", "query_text": "What happened to the Millbrook bridge after the flood?", "difficulty": "medium"},
    ],
    "crawford_roses": [
        {"state_id": "a", "query_text": "Describe Mrs. Crawford's rose garden in its prime.", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to Mrs. Crawford's rose garden after the disease struck?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What was Mrs. Crawford's garden like after it was restored?", "difficulty": "high"},
    ],
    "foxworth_gates": [
        {"state_id": "a", "query_text": "What did the gates of Foxworth Hall look like in their original state?", "difficulty": "low"},
        {"state_id": "b", "query_text": "What happened to the Foxworth estate gates after the original ones were removed?", "difficulty": "medium"},
    ],
    "aldridge_chest": [
        {"state_id": "a", "query_text": "What was Captain Aldridge's sea chest like while it remained sealed?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What was found when Captain Aldridge's sea chest was opened after his death?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What remained in Captain Aldridge's sea chest at the very end?", "difficulty": "high"},
    ],
    "thornwood_window": [
        {"state_id": "a", "query_text": "Describe the stained glass window in Thornwood church as it was originally.", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to the Thornwood church window after the storm?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "How did the restored Thornwood church window differ from its original form?", "difficulty": "high"},
    ],
    "pemberton_journal": [
        {"state_id": "a", "query_text": "What was Dr. Pemberton's medical journal like before it was damaged?", "difficulty": "low"},
        {"state_id": "b", "query_text": "What became of Dr. Pemberton's medical journal after the flood?", "difficulty": "medium"},
    ],
    "kingscross_bell": [
        {"state_id": "a", "query_text": "Describe the Kingscross market bell in its working days.", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to the Kingscross market bell after it developed a fault?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What became of the Kingscross market bell after it was recast?", "difficulty": "high"},
    ],
    "pennington_tapestry": [
        {"state_id": "a", "query_text": "What did the Pennington tapestry depict in its original condition?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What was the state of the Pennington tapestry after it was found in storage?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "How did the restored Pennington tapestry differ from the original?", "difficulty": "high"},
    ],
    "ashbury_thunder": [
        {"state_id": "a", "query_text": "What was Lord Ashbury's stallion Thunder like in his racing prime?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to Thunder after the racing accident?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What became of Thunder in his later years?", "difficulty": "high"},
    ],
    "fairweather_musicbox": [
        {"state_id": "a", "query_text": "Describe the Fairweather music box when it was still in use.", "difficulty": "low"},
        {"state_id": "b", "query_text": "What condition was the Fairweather music box in after it fell into disrepair?", "difficulty": "medium"},
    ],
    "thornwood_well": [
        {"state_id": "a", "query_text": "What was the village well at Thornwood like when it was still in use?", "difficulty": "low"},
        {"state_id": "b", "query_text": "What happened to the Thornwood well after the bucket was lost?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What was the final state of the Thornwood village well?", "difficulty": "high"},
    ],
    "templewater_painting": [
        {"state_id": "a", "query_text": "Describe Miss Templewater's painting as it was first created.", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to Miss Templewater's original painting?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What remnant of Miss Templewater's original painting was preserved?", "difficulty": "high"},
    ],
    "millbrook_wall": [
        {"state_id": "a", "query_text": "What was the Millbrook rectory garden wall like originally?", "difficulty": "low"},
        {"state_id": "b", "query_text": "What happened to the Millbrook rectory wall after the dispute?", "difficulty": "medium"},
    ],
    "troupe_wagon": [
        {"state_id": "a", "query_text": "What did the Harper troupe's wagon look like in its original form?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to the Harper troupe's wagon while it sat by the roadside?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "How did the Harper troupe's wagon appear after it was repaired?", "difficulty": "high"},
    ],
    "ravenscroft_tea": [
        {"state_id": "a", "query_text": "Describe the Ravenscroft silver tea service in its complete state.", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What happened to the Ravenscroft tea service during the family's move?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "How was the Ravenscroft tea service restored after the loss?", "difficulty": "high"},
    ],
    "foxworth_oak": [
        {"state_id": "a", "query_text": "What was the ancient oak at Foxworth field like before it was struck?", "difficulty": "low"},
        {"state_id": "b", "query_text": "What happened to the Foxworth field oak after being struck?", "difficulty": "medium"},
    ],
    "whitmore_trunk": [
        {"state_id": "a", "query_text": "What did Mr. Whitmore's traveling trunk contain when he returned from his Grand Tour?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What did Mr. Whitmore pack in his traveling trunk for his next journey?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What did Mr. Whitmore's traveling trunk hold in its final state?", "difficulty": "high"},
    ],
    "ashbury_hermitage": [
        {"state_id": "a", "query_text": "What was the hermitage in Ashbury Park like when it was first built?", "difficulty": "medium"},
        {"state_id": "b", "query_text": "What condition was the Ashbury Park hermitage in after years of neglect?", "difficulty": "medium"},
        {"state_id": "c", "query_text": "What did the Ashbury Park hermitage become after it was repurposed?", "difficulty": "high"},
    ],
}


# ---------------------------------------------------------------------------
# Novel chunking and insertion logic
# ---------------------------------------------------------------------------

def chunk_text(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> list[dict]:
    """Split novel text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 50:
            break
        chunks.append({
            "position": i,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words),
        })
        if i + chunk_size >= len(words):
            break
    return chunks


def distribute_planted_chunks(
    n_original: int, scenarios: list[dict]
) -> list[tuple[str, str, int]]:
    """Assign insertion positions for planted state paragraphs.

    Returns list of (scenario_id, state_id, insert_after_position).
    States from the same scenario are placed far apart by interleaving:
    position 0 = all scenarios' state A, position 1 = all scenarios' state B, etc.
    """
    # Group states by scenario
    by_scenario: dict[str, list[str]] = {}
    for sc in scenarios:
        by_scenario[sc["id"]] = [st["id"] for st in sc["states"]]

    max_states = max(len(states) for states in by_scenario.values())
    n_scenarios = len(scenarios)

    # Build interleaved assignment list:
    # First all scenario state_a's, then all state_b's, then all state_c's
    ordered: list[tuple[str, str]] = []
    for state_idx in range(max_states):
        for sc in scenarios:
            states = by_scenario[sc["id"]]
            if state_idx < len(states):
                ordered.append((sc["id"], states[state_idx]))

    n_planted = len(ordered)
    spacing = max(1, n_original // (n_planted + 1))

    # Create evenly spaced positions
    positions = [spacing * (i + 1) for i in range(n_planted)]
    positions = [min(p, n_original - 1) for p in positions]

    result = []
    for i, (sc_id, st_id) in enumerate(ordered):
        result.append((sc_id, st_id, positions[i]))

    return result


def validate_scenarios(scenarios: list[dict], novel_text: str) -> list[str]:
    """Validate scenarios against the 5 criteria. Returns list of issues."""
    issues = []

    novel_lower = novel_text.lower()

    for sc in scenarios:
        sc_id = sc["id"]
        states = sc["states"]
        questions = sc["questions"]

        # Criterion 1: states are distinguishable
        state_texts = [st["paragraph"].lower() for st in states]
        for i, t1 in enumerate(state_texts):
            for j, t2 in enumerate(state_texts):
                if i < j:
                    # Check that states differ by more than trivial amount
                    words1 = set(t1.split())
                    words2 = set(t2.split())
                    unique_to_1 = words1 - words2
                    unique_to_2 = words2 - words1
                    if len(unique_to_1) < 5 or len(unique_to_2) < 5:
                        issues.append(
                            f"{sc_id} states {states[i]['id']}/{states[j]['id']}: "
                            f"too similar (unique words: {len(unique_to_1)}, {len(unique_to_2)})"
                        )

        # Criterion 2: at least 2 states per scenario
        if len(states) < 2:
            issues.append(f"{sc_id}: only {len(states)} states (need >= 2)")

        # Criterion 3: no position/chapter hints in questions
        for q in questions:
            qt = q["query_text"].lower()
            for forbidden in ["chapter", "section ", "page ", "volume", "book "]:
                if forbidden in qt:
                    issues.append(f"{sc_id} q:{q['state_id']}: '{forbidden}' in query")

        # Criterion 4: planted text doesn't overlap with novel
        for st in states:
            # Check if a significant phrase from the paragraph appears in the novel
            # Use first sentence as a proxy
            first_sentence = st["paragraph"].split(".")[0].strip().lower()
            words = first_sentence.split()
            if len(words) >= 6:
                phrase = " ".join(words[:6])
                if phrase in novel_lower:
                    issues.append(
                        f"{sc_id} state {st['id']}: first 6 words match novel text"
                    )

        # Each question should have a matching state
        state_ids = {st["id"] for st in states}
        for q in questions:
            if q["state_id"] not in state_ids:
                issues.append(f"{sc_id}: question references unknown state {q['state_id']}")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    novel_path = DATA_DIR / "benchmark_narrative" / "pride_and_prejudice.txt"
    print(f"Reading novel: {novel_path}")
    with open(novel_path, encoding="utf-8") as f:
        novel_text = f.read()
    print(f"  Novel: {len(novel_text):,} chars, ~{len(novel_text.split())} words")

    # Validate scenarios
    print("\nValidating scenarios against criteria...")
    issues = validate_scenarios(SCENARIOS, novel_text)
    if issues:
        print(f"  Found {len(issues)} issues:")
        for issue in issues[:20]:
            print(f"    - {issue}")
        print()
    else:
        print("  All scenarios pass validation")

    # Count stats
    n_scenarios = len(SCENARIOS)
    n_states = sum(len(sc["states"]) for sc in SCENARIOS)
    n_questions = sum(len(sc["questions"]) for sc in SCENARIOS)
    print(f"\n  Scenarios: {n_scenarios}")
    print(f"  State paragraphs: {n_states}")
    print(f"  Questions: {n_questions}")

    # Chunk the novel
    print("\nChunking novel...")
    original_chunks = chunk_text(novel_text, chunk_size=512, overlap=64)
    print(f"  Original chunks: {len(original_chunks)}")

    # Distribute planted chunks
    print("Distributing planted paragraphs...")
    assignments = distribute_planted_chunks(len(original_chunks), SCENARIOS)
    print(f"  Assignments: {len(assignments)}")

    # Build planted chunk lookup: (scenario_id, state_id) -> state paragraph
    state_lookup: dict[tuple[str, str], str] = {}
    for sc in SCENARIOS:
        for st in sc["states"]:
            state_lookup[(sc["id"], st["id"])] = st["paragraph"]

    # Build all chunks: original + planted, interleaved
    print("Building final chunk database...")
    all_chunks: list[dict] = []

    # Create a mapping from assignment position to planted chunks
    planted_by_pos: dict[int, list[tuple[str, str, str]]] = {}
    for sc_id, st_id, pos in assignments:
        paragraph = state_lookup[(sc_id, st_id)]
        planted_by_pos.setdefault(pos, []).append((sc_id, st_id, paragraph))

    # Track planted chunk IDs for question resolution
    planted_chunk_map: dict[tuple[str, str], int] = {}
    chunk_counter = 0

    for i, oc in enumerate(original_chunks):
        # Insert any planted chunks that go before this position
        if i in planted_by_pos:
            for sc_id, st_id, paragraph in planted_by_pos[i]:
                all_chunks.append({
                    "chunk_id": chunk_counter,
                    "type": "planted",
                    "scenario_id": sc_id,
                    "state_id": st_id,
                    "text": paragraph,
                    "insert_after_original_chunk": i,
                })
                planted_chunk_map[(sc_id, st_id)] = chunk_counter
                chunk_counter += 1

        # Add original chunk
        all_chunks.append({
            "chunk_id": chunk_counter,
            "type": "original",
            "text": oc["text"],
            "position": oc["position"],
            "word_count": oc["word_count"],
        })
        chunk_counter += 1

    # Handle any remaining planted chunks at the end
    if len(original_chunks) in planted_by_pos:
        for sc_id, st_id, paragraph in planted_by_pos[len(original_chunks)]:
            all_chunks.append({
                "chunk_id": chunk_counter,
                "type": "planted",
                "scenario_id": sc_id,
                "state_id": st_id,
                "text": paragraph,
                "insert_after_original_chunk": len(original_chunks),
            })
            planted_chunk_map[(sc_id, st_id)] = chunk_counter
            chunk_counter += 1

    n_planted = sum(1 for c in all_chunks if c["type"] == "planted")
    n_original = sum(1 for c in all_chunks if c["type"] == "original")
    print(f"  Total chunks: {len(all_chunks)} ({n_original} original + {n_planted} planted)")

    # Verify all planted chunks are assigned
    for sc in SCENARIOS:
        for st in sc["states"]:
            if (sc["id"], st["id"]) not in planted_chunk_map:
                print(f"  WARNING: {sc['id']}/{st['id']} not assigned to any chunk!")

    # Build questions (specific + generic)
    print("\nGenerating questions...")
    questions = []
    for sc in SCENARIOS:
        gold_state_ids = [st["id"] for st in sc["states"]]

        all_qs = [(q, "specific") for q in sc["questions"]]
        all_qs += [(q, "generic") for q in GENERIC_QUESTIONS.get(sc["id"], [])]

        for q, style in all_qs:
            gold_key = (sc["id"], q["state_id"])
            if gold_key not in planted_chunk_map:
                print(f"  WARNING: No chunk for {gold_key}")
                continue

            # Build state summary from gold paragraph
            gold_para = state_lookup[gold_key]
            summary = gold_para[:120].replace("\n", " ").strip()

            qid = f"q_planted_{sc['id']}_{q['state_id']}_{style}"
            questions.append({
                "id": qid,
                "type": "single_version",
                "difficulty": q["difficulty"],
                "question_style": style,
                "query_text": q["query_text"],
                "gold_scenario": sc["id"],
                "gold_state": q["state_id"],
                "distractor_states": [
                    sid for sid in gold_state_ids if sid != q["state_id"]
                ],
                "gold_state_summary": summary,
                "dynamic_top_k": 1,
            })

    print(f"  Generated {len(questions)} questions")

    # Distribution by style
    by_style: dict[str, int] = {}
    for q in questions:
        by_style[q["question_style"]] = by_style.get(q["question_style"], 0) + 1
    print("  By style:")
    for style, count in sorted(by_style.items()):
        print(f"    {style}: {count}")

    # Difficulty distribution
    by_diff: dict[str, int] = {}
    for q in questions:
        by_diff[q["difficulty"]] = by_diff.get(q["difficulty"], 0) + 1
    print("  By difficulty:")
    for diff, count in sorted(by_diff.items()):
        print(f"    {diff}: {count}")

    # Save novel_chunks.jsonl
    chunks_path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_chunks)} chunks to {chunks_path}")

    # Save questions.jsonl
    questions_path = DATA_DIR / "benchmark_narrative" / "questions.jsonl"
    with open(questions_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved {len(questions)} questions to {questions_path}")


if __name__ == "__main__":
    main()
