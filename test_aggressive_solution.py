import json
import torch
import argparse
import re
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from moviepy import VideoFileClip
from transformers import AutoModelForCausalLM

MODEL_PATH = "AIDC-AI/Ovis2.5-9B"
VIDEO_DIR = "/home/ra164195/aggressive_behavior_project/videos/"
NUM_FRAMES = 8
SAVE_EVERY = 10  # save checkpoint every N videos

avoid_videos = ["aggressive__talking_1_trim_0.mp4"]

# --- Ovis2.5 generation settings ---
max_pixels = 896 * 896  # As per Ovis2.5 recommendation
max_new_tokens = 64
thinking_budget = 2048
enable_thinking = False
enable_thinking_budget = False

# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--questions_json", type=str, help="Path to pre-generated questions JSON")
parser.add_argument("--start_index", type=int, default=0, help="Video index to start from (0-based)")
parser.add_argument("--end_index", type=int, default=None, help="Video index to end at, exclusive (0-based). If not specified, runs to the end.")
parser.add_argument("--template", type=str, default="role_graph",
                    help="Which prompt template to use")
args = parser.parse_args()

# ---------- sanity check ----------
if args.end_index is not None and args.end_index <= args.start_index:
    raise ValueError(f"end_index ({args.end_index}) must be greater than start_index ({args.start_index})")

QUESTIONS_JSON = args.questions_json
end_suffix = f"_to{args.end_index}" if args.end_index is not None else ""
OUTPUT_JSON = f"ovis2_5_9B_{Path(QUESTIONS_JSON).stem}_from{args.start_index}{end_suffix}_{args.template}.json"

# ---------- prompt templates ----------
QUESTION_TEMPLATES = {

   "role_graph": (
        r"""
        Your primary task is to act as a causal reasoning engine for understanding aggressive scenes in videos. You will observe the persons present and use a causal graph to identify each person's role and the nature of the aggressive interaction.

        ### Component Definitions

        First, understand the meaning of each component (node) in our causal graph. These are the building blocks we use to analyze any aggressive scene.

        * P (Persons): All individuals visible in the video. For example, 'a man in a black coat', 'a woman in a red shirt', 'three people in the background'. Each person must be assigned exactly one role.
        * I (Interaction): Whether a given person is engaged in a direct physical interaction with another person. This is a binary branch point:
            - YES — the person is physically engaged with at least one other person (e.g., striking, grabbing, restraining, being struck).
            - NO — the person has no physical contact with any other person in the scene.
        * D (Direction): For persons where I = YES, the direction of the aggressive interaction:
            - INITIATING — the person is the one delivering the aggressive act (e.g., throwing a punch, grabbing, pushing).
            - RECEIVING — the person is the one on the receiving end of the aggressive act (e.g., being punched, being grabbed, being pushed).
        * R (Role): The role assigned to each person, inferred from I and D:
            - Aggressor — a person where I = YES and D = INITIATING.
            - Victim — a person where I = YES and D = RECEIVING.
            - Bystander — a person where I = NO (not physically involved).
        * A (Aggressive Action): The specific type of aggressive behavior being performed, inferred from the Aggressor's motions and the Victim's reactions. For example, 'punch', 'shove', 'kick', 'choke', 'hair grab'. This is determined jointly by the Aggressor and Victim roles.

        ---

        ### The Causal Graph: How Components are Related

        The components are connected in a Directed Acyclic Graph (DAG). Each arrow (→) represents a direction of inference — follow the graph in order to assign roles correctly.

        The directed edges of the graph are:
        P → I
        I (YES) → D
        I (NO) → R (Bystander)
        D (INITIATING) → R (Aggressor)
        D (RECEIVING) → R (Victim)
        R (Aggressor) + R (Victim) → A

        Here are the causal relationships explained in detail:

        * `P → I` (Does this person interact?): For every person identified in P, ask whether they are physically interacting with another person. This is the first branch point. A person standing and watching is I = NO. A person throwing a punch is I = YES.
        * `I (YES) → D` (What is the direction of their interaction?): For persons with I = YES, determine whether they are initiating or receiving the physical contact. Watch carefully — the aggressor moves *toward* the victim with force; the victim reacts *to* the aggressor's action.
        * `I (NO) → R (Bystander)`: Any person with no physical involvement is immediately classified as a Bystander. No further reasoning is needed for them.
        * `D (INITIATING) → R (Aggressor)`: A person who is initiating the aggressive physical contact is the Aggressor.
        * `D (RECEIVING) → R (Victim)`: A person who is on the receiving end of the aggressive physical contact is the Victim.
        * `R (Aggressor) + R (Victim) → A` (What is the Aggressive Action?): Only after both the Aggressor and Victim are identified can the Aggressive Action be determined. Observe the Aggressor's body motion (e.g., arm swinging forward) and the Victim's reaction (e.g., head snapping back) together to name the action precisely.

        ---

        ### Illustrative Examples

        #### Example 1: A clear two-person aggressive scene
        A video shows two people. One person in a black jacket swings their fist at a person in a white shirt, who stumbles backward.

        * Causal Analysis (internal, do not output):
            1. P: Two persons — 'person in a black jacket', 'person in a white shirt'.
            2. I for 'person in a black jacket': YES (physical contact with fist).
               I for 'person in a white shirt': YES (receiving the fist).
            3. D for 'person in a black jacket': INITIATING (swinging the fist).
               D for 'person in a white shirt': RECEIVING (stumbling from the impact).
            4. R: 'person in a black jacket' → Aggressor. 'person in a white shirt' → Victim.
            5. A: The Aggressor's motion (arm swinging forward, fist connecting) and Victim's reaction (head snapping, stumbling) → Aggressive Action = 'punch'.

        #### Example 2: A scene with a bystander
        A video shows three people. One person in a red hoodie grabs another person in a grey jacket by the collar. A third person in a blue shirt stands nearby watching.

        * Causal Analysis (internal, do not output):
            1. P: Three persons — 'person in a red hoodie', 'person in a grey jacket', 'person in a blue shirt'.
            2. I for 'person in a red hoodie': YES. I for 'person in a grey jacket': YES. I for 'person in a blue shirt': NO.
            3. D for 'person in a red hoodie': INITIATING (grabbing). D for 'person in a grey jacket': RECEIVING (being grabbed).
            4. R: 'person in a red hoodie' → Aggressor. 'person in a grey jacket' → Victim. 'person in a blue shirt' → Bystander (I = NO, skip D).
            5. A: The Aggressor grabs the Victim's collar and pulls — Aggressive Action = 'clothing grab'.

        #### Example 3: No aggressive action
        A video shows two people standing and talking to each other without any physical contact.

        * Causal Analysis (internal, do not output):
            1. P: Two persons.
            2. I for both persons: NO (no physical contact with any other person).
            3. Since I = NO for all persons, no one reaches the D step.
            4. R: Both persons → Bystander (no aggressor, no victim).
            5. A: No Aggressor and no Victim exist → No aggressive action is taking place.

        ---

        ### Your Task

        You will now be given a new video. Apply the causal graph above to reason about every person you see.

        Reason through these steps internally. Do not output your reasoning.
        * P (Persons): Identify all individuals in the video. For each person, carefully describe their appearance — focus first on clothing (color and type, e.g., 'black coat', 'white t-shirt', 'blue jeans'), then hair color/style, then any other distinguishing feature. These descriptions will serve as each person's identifier throughout the rest of your reasoning, and must match the appearance-based labels used in the answer options.
        * I (Interaction): For each person, determine YES or NO — are they physically interacting with another person?
        * D (Direction): For each person with I = YES, determine INITIATING or RECEIVING.
        * R (Role): Assign each person a role — Aggressor, Victim, or Bystander — following the graph edges.
        * A (Aggressive Action): Using the identified Aggressor and Victim, determine the specific aggressive action. If no Aggressor and Victim exist, the answer is that no aggressive action is taking place.

        From the list of options below, select the answer that most accurately reflects the question asked about the video.
        There might also a choice if none of the other options apply.
        Review the video and options carefully. Only one choice is correct.

        Output format: reply with the choice number only.
        """
    ),
}

# ---------- prompt formatting ----------
def format_prompt(prompt: str, answers: list[str], template: str) -> str:
    system_instruction = QUESTION_TEMPLATES[template].strip()
    lines = [
        system_instruction,
        "",
        "---",
        "",
        f"Question: {prompt}",
        "",
        "Options:",
    ]
    for i, answer in enumerate(answers):
        lines.append(f"{i + 1}. {answer}")
    lines.extend([
        "",
        "Answer with ONLY the option number (e.g., '1' or '2').",
    ])
    return "\n".join(lines)


# ---------- load model ----------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).cuda()
model.eval()

# ---------- video frame sampling (Ovis2.5 style) ----------
def sample_video_frames(video_path, num_frames=NUM_FRAMES):
    with VideoFileClip(str(video_path)) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            stride = total_frames / num_frames
            indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
        frames = [Image.fromarray(clip.get_frame(i / clip.fps)) for i in indices]
        return frames

# ---------- strip <think>…</think> from response ----------
def split_think_and_answer(text):
    pattern = re.compile(r'(<think>.*?</think>)', re.DOTALL)
    match = pattern.search(text)
    if match:
        think_part = match.group(1)
        rest = text[:match.start()] + text[match.end():]
        return think_part, rest.strip()
    else:
        return None, text

# ---------- load questions ----------
with open(QUESTIONS_JSON, "r") as f:
    data = json.load(f)
questions_by_video = data["questions_by_video"]
#questions_by_video = dict(list(data["questions_by_video"].items())[:10])  # limit to 10 videos for testing
print(f"Loaded questions for {len(questions_by_video)} videos")
print(f"Template: {args.template}")
print(f"Output will be saved to: {OUTPUT_JSON}")

# ---------- apply start/end index ----------
all_video_names = list(questions_by_video.keys())
if args.start_index > 0:
    print(f"Resuming from video index {args.start_index} ({all_video_names[args.start_index]})")
if args.end_index is not None:
    print(f"Running up to video index {args.end_index} (exclusive)")
questions_by_video = dict(list(questions_by_video.items())[args.start_index:args.end_index])

# ---------- checkpoint save helper ----------
def save_checkpoint(results: list, output_path: str):
    output = {
        "total_questions": len(results),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [Checkpoint] Saved {len(results)} results to {output_path}")

# ---------- evaluate ----------
all_results = []
video_dir = Path(VIDEO_DIR)

for video_idx, (video_name, questions) in enumerate(tqdm(questions_by_video.items(), desc="Videos", unit="video")):
    video_path = video_dir / video_name
    if not video_path.exists():
        print(f"Skipping {video_name} — file not found")
        continue

    if video_name in avoid_videos:
        print(f"Skipping {video_name} as it's in the avoid list.")
        continue

    print(f"\nProcessing: {video_name} ({len(questions)} questions)")

    for q in questions:
        formatted_prompt = format_prompt(q["prompt"], q["answers"], args.template)

        try:
            frames = sample_video_frames(video_path, num_frames=NUM_FRAMES)

            # Ovis2.5: preprocess_inputs handles all preprocessing internally
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": formatted_prompt}
                    ],
                }
            ]

            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                max_pixels=max_pixels,
                enable_thinking=enable_thinking,
            )

            input_ids = input_ids.cuda()
            pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
            grid_thws = grid_thws.cuda() if grid_thws is not None else None

        except ValueError as e:
            print(f"Skipping {video_name} — {e}")
            break  # skip all questions for this video

        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking_budget,
                thinking_budget=thinking_budget,
                eos_token_id=model.text_tokenizer.eos_token_id,
                pad_token_id=model.text_tokenizer.pad_token_id,
            )[0]

        response = model.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        _, model_response = split_think_and_answer(response)
        model_response = model_response.strip()

        result = {
            "video_name":     q["video_name"],
            "question_type":  q["question_type"],
            "prompt":         q["prompt"],
            "answers":        q["answers"],
            "correct_answer": q["correct_answer"],
            "correct_index":  q["correct_index"],  # 0-based
            "model_response": model_response,       # 1-based number string e.g. "3"
            "template":       args.template,        # track which template was used
        }
        #print(result)
        all_results.append(result)

    # save checkpoint every SAVE_EVERY videos
    if (video_idx + 1) % SAVE_EVERY == 0:
        save_checkpoint(all_results, OUTPUT_JSON)

# ---------- save output ----------
output = {
    "total_questions": len(all_results),
    "template": args.template,
    "results": all_results,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone! {len(all_results)} questions saved to {OUTPUT_JSON}")