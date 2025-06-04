#!/usr/bin/env python3
"""
AI LIFE SIMULATION - Living Character with Learning Memory
Author: MERT YANDIMATA

Uses Qwen3-4B-4bit (via MLX-LM) to simulate a character living through days with dynamic, realistic scenarios.
Each run = 1 day (24 hours), learns from consequences, handles obstacles autonomously, and checks inventory before deciding.
This version runs in English, includes random positive and negative twists, and has an “author” note at the top.
"""

import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Attempt to import MLX-LM for local inference
try:
    import mlx_lm
    MLX_AVAILABLE = True
    print("✅ MLX-LM available for local inference")
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX-LM not available - install with: pip install mlx-lm")


class AILifeSimulation:
    """
    Class to simulate an AI-driven character living through days, making decisions,
    recording memories, evolving personality, and learning from outcomes.
    The AI “becomes” the character: first-person informal English, checks inventory, and
    experiences random good/bad events within each hourly scenario.
    """

    def __init__(self, character_name: str = "Alex"):
        """
        Initialize the simulation: set up file paths, load or create data, load ML model.
        """
        self.character_name: str = character_name
        self.data_dir: Path = Path("ai_life_data")
        self.data_dir.mkdir(exist_ok=True)

        # File paths for persistent storage
        self.character_file: Path = self.data_dir / f"{character_name}_character.json"
        self.memories_file: Path = self.data_dir / f"{character_name}_memories.json"
        self.relationships_file: Path = self.data_dir / f"{character_name}_relationships.json"
        self.daily_logs_file: Path = self.data_dir / f"{character_name}_daily_logs.json"
        self.learning_file: Path = self.data_dir / f"{character_name}_learning.json"

        # Load or initialize data structures
        self.character: dict = self.load_or_create_character()
        self.memories: dict = self.load_or_create_memories()
        self.relationships: dict = self.load_or_create_relationships()
        self.daily_logs: dict = self.load_or_create_daily_logs()
        self.learning_data: dict = self.load_or_create_learning()

        # Load the Qwen3-4B-4bit model if available
        self.load_model()

        # Summary printout
        print(f"🌟 AI Life Simulation initialized for {self.character_name}")
        print(f"📅 Current day: {self.character['current_day']}")
        print(f"💭 Memories: {len(self.memories['significant_events'])} events")
        print(f"👥 Relationships: {len(self.relationships)} people")

    def load_model(self) -> None:
        """
        Load Qwen3-4B-4bit model via mlx_lm if available; otherwise set model to None.
        """
        if not MLX_AVAILABLE:
            print("❌ Cannot load model - MLX-LM not available")
            self.model = None
            self.tokenizer = None
            return

        try:
            print("🔄 Loading Qwen3-4B-4bit model...")
            self.model, self.tokenizer = mlx_lm.load("mlx-community/Qwen3-4B-4bit")
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def load_or_create_character(self) -> dict:
        """
        Load character state from file if it exists; otherwise create a new character.
        Returns a dictionary with character attributes, including inventory.
        """
        if self.character_file.exists():
            with open(self.character_file, 'r') as f:
                data = json.load(f)
                return data
        else:
            # Create a new character with random initial stats and inventory
            initial_character = {
                "name": self.character_name,
                "current_day": 1,
                "age": random.randint(22, 35),
                "personality": {
                    "openness": random.uniform(3, 9),
                    "conscientiousness": random.uniform(3, 9),
                    "extraversion": random.uniform(2, 9),
                    "agreeableness": random.uniform(3, 9),
                    "neuroticism": random.uniform(2, 8),
                    "spontaneity": random.uniform(3, 8),
                    "ambition": random.uniform(3, 8),
                    "creativity": random.uniform(3, 9)
                },
                "current_stats": {
                    "happiness": random.uniform(4, 7),
                    "energy": random.uniform(5, 8),
                    "social_need": random.uniform(3, 8),
                    "stress": random.uniform(2, 6),
                    "health": random.uniform(6, 9),
                    "motivation": random.uniform(4, 7),
                    "curiosity": random.uniform(4, 8),
                    "loneliness": random.uniform(2, 6)
                },
                "current_situation": "Trying to figure out what matters",
                "location": "home",
                "occupation": self.generate_random_occupation(),
                "core_desire": "To feel genuinely happy and fulfilled",
                "current_goals": [],
                "inventory": {
                    "skincare_kit": random.choice([0, 1]),  # 0 = none, 1 = has it
                    "money": random.randint(0, 50),         # in dollars
                    "food_items": ["bread", "cheese"] if random.random() < 0.7 else [],
                    "phone_credit": random.randint(0, 20)   # dollars of credit
                },
                "personality_evolution": {
                    "changes": [],
                    "last_major_shift": None
                }
            }
            return initial_character

    def load_or_create_memories(self) -> dict:
        """
        Load memories data from file if it exists; otherwise create an empty memories structure.
        """
        if self.memories_file.exists():
            with open(self.memories_file, 'r') as f:
                data = json.load(f)
                return data
        else:
            initial_memories = {
                "significant_events": [],
                "recent_actions": [],
                "learned_patterns": {},
                "emotional_memories": [],
                "daily_summaries": []
            }
            return initial_memories

    def generate_random_occupation(self) -> str:
        """
        Select a random occupation for the character from a predefined list.
        """
        occupations = [
            "freelance writer", "coffee shop barista", "graphic designer",
            "yoga instructor", "bookstore clerk", "photographer",
            "musician", "social worker", "teacher", "chef",
            "artist", "therapist", "librarian", "researcher",
            "software developer", "marketing coordinator", "nurse"
        ]
        return random.choice(occupations)

    def load_or_create_relationships(self) -> dict:
        """
        Load relationships data from file if it exists; otherwise create default relationships.
        """
        if self.relationships_file.exists():
            with open(self.relationships_file, 'r') as f:
                data = json.load(f)
                return data
        else:
            initial_relationships = {
                "Sarah": {
                    "type": "best_friend",
                    "closeness": 8.2,
                    "last_contact": "3_days_ago",
                    "personality": "caring, funny, supportive",
                    "history": "Met in college, stayed close friends",
                    "interaction_count": 0
                },
                "Mom": {
                    "type": "family",
                    "closeness": 9.1,
                    "last_contact": "1_week_ago",
                    "personality": "loving, worried, wise",
                    "history": "Always been supportive",
                    "interaction_count": 0
                },
                "Jake": {
                    "type": "coworker",
                    "closeness": 6.5,
                    "last_contact": "yesterday",
                    "personality": "professional, friendly, smart",
                    "history": "Work colleague, potential friend",
                    "interaction_count": 0
                }
            }
            return initial_relationships

    def load_or_create_daily_logs(self) -> dict:
        """
        Load daily logs from file if it exists; otherwise create an empty logs structure.
        """
        if self.daily_logs_file.exists():
            with open(self.daily_logs_file, 'r') as f:
                data = json.load(f)
                return data
        else:
            initial_logs = {"logs": []}
            return initial_logs

    def load_or_create_learning(self) -> dict:
        """
        Load learning data from file if it exists; otherwise create an empty learning structure.
        """
        if self.learning_file.exists():
            with open(self.learning_file, 'r') as f:
                data = json.load(f)
                return data
        else:
            initial_learning = {
                "action_outcomes": {},
                "pattern_recognition": {},
                "relationship_learning": {},
                "mood_triggers": {},
                "successful_strategies": [],
                "failed_strategies": []
            }
            return initial_learning

    def save_all_data(self) -> None:
        """
        Save all data structures (character, memories, relationships, logs, learning) to their respective JSON files.
        """
        with open(self.character_file, 'w') as f:
            json.dump(self.character, f, indent=2)

        with open(self.memories_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

        with open(self.relationships_file, 'w') as f:
            json.dump(self.relationships, f, indent=2)

        with open(self.daily_logs_file, 'w') as f:
            json.dump(self.daily_logs, f, indent=2)

        with open(self.learning_file, 'w') as f:
            json.dump(self.learning_data, f, indent=2)

    def dynamic_goal_generation(self) -> list:
        """
        Generate 2-3 goals for the character based on current happiness, loneliness, and stress.
        Includes recent context and learning context so the model can pick relevant goals.
        """
        stats = self.character['current_stats']
        happiness = stats['happiness']
        loneliness = stats['loneliness']
        stress = stats['stress']

        context = (
            f"I am {self.character['name']}. Right now I feel:\n"
            f"- Happiness: {happiness:.1f}/10\n"
            f"- Loneliness: {loneliness:.1f}/10\n"
            f"- Stress: {stress:.1f}/10\n"
            f"- Core desire: {self.character['core_desire']}\n"
            f"- Inventory: {self.character['inventory']}\n"
            f"- Recent actions: {self.get_recent_context()}\n"
            f"- Learned so far: {self.get_learning_context()}\n\n"
            "Now, as if you are this character in first-person informal English, "
            "generate 2-3 personal goals that feel meaningful right now. "
            "Check your inventory; if you lack something, plan how to get it. "
            "Write goals as bullets with '-', using casual real-life language, "
            "e.g. “- I should grab a bite because I’m starving,” or “- Need to check Netflix for a new show.” "
            "No extra explanation—just the bullet points.\n"
        )

        if self.model and self.tokenizer:
            response = self.generate_response(context, max_tokens=2000)
            goals = self.extract_goals_from_response(response)
        else:
            goals = self.generate_fallback_goals()

        return goals

    def extract_goals_from_response(self, response: str) -> list:
        """
        Parse the AI model's response to extract goal lines.
        Lines starting with '-' or containing the word 'goal' are considered.
        """
        lines = response.split('\n')
        goals = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('-') or 'goal' in stripped.lower():
                goal_text = stripped.lstrip('-').strip()
                if len(goal_text) > 10:
                    goals.append(goal_text)
        if not goals:
            goals = self.generate_fallback_goals()
        return goals[:3]

    def generate_fallback_goals(self) -> list:
        """
        Return a random sample of 2 fallback goals when the model cannot generate goals.
        """
        possible_goals = [
            "- I should find something to eat because I’m really hungry",
            "- Maybe text Sarah to see if she wants to hang out",
            "- Try a short walk outside to clear my head",
            "- Work on a small project to feel productive",
            "- Listen to music to lift my mood",
            "- Take a quick nap if I’m too tired",
            "- Call Mom to check in",
            "- Read a bit of a book to relax",
            "- Do a brief workout because I feel restless",
            "- Try cooking something new since I’ve got ingredients"
        ]
        return random.sample(possible_goals, 2)

    def evolve_personality(self, significant_experience: str) -> None:
        """
        With a 10% chance, adjust one of the character's personality traits based on whether
        the experience was positive or negative. Record the change in personality_evolution.
        """
        if random.random() < 0.1:
            trait_to_change = random.choice(list(self.character['personality'].keys()))
            current_value = self.character['personality'][trait_to_change]

            if 'positive' in significant_experience.lower() or 'happy' in significant_experience.lower():
                change_amount = random.uniform(0.1, 0.3)
            else:
                change_amount = random.uniform(-0.3, -0.1)

            new_value = max(1, min(10, current_value + change_amount))
            self.character['personality'][trait_to_change] = new_value

            change_record = {
                "day": self.character['current_day'],
                "trait": trait_to_change,
                "change": change_amount,
                "trigger": significant_experience,
                "new_value": new_value
            }
            self.character['personality_evolution']['changes'].append(change_record)

            print(f"🧠 Personality shift: {trait_to_change} changed by {change_amount:.2f} due to '{significant_experience}'")

    def dynamic_goal_update(self) -> None:
        """
        Update current_goals if:
        - It's been 3 days since last update, OR
        - Happiness is below threshold, OR
        - There are no current goals.
        """
        day = self.character['current_day']
        happiness = self.character['current_stats']['happiness']
        current_goals = self.character['current_goals']

        should_update = (
            day % 3 == 0 or
            happiness < 4 or
            len(current_goals) == 0
        )
        if should_update:
            new_goals = self.dynamic_goal_generation()
            self.character['current_goals'] = new_goals
            print(f"🎯 New goals: {new_goals}")

    def get_time_period(self, hour: int) -> str:
        """
        Return a string description of the time of day given an hour.
        """
        if 6 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 17:
            return "Afternoon"
        elif 18 <= hour <= 22:
            return "Evening"
        else:
            return "Night"

    def get_recent_context(self) -> str:
        """
        Return a short summary of the last 3 recent actions (if any) for use as context.
        """
        if not self.memories['recent_actions']:
            return "fresh start"
        recent = self.memories['recent_actions'][-3:]
        summaries = [action['summary'] for action in recent]
        return " → ".join(summaries)

    def get_learning_context(self) -> str:
        """
        Return a summary of the last two successful strategies for context.
        """
        successful = self.learning_data['successful_strategies'][-2:] if self.learning_data['successful_strategies'] else []
        if successful:
            return f"I learned that: {', '.join(successful)}"
        else:
            return "Still figuring out what works"

    def happiness_focused_prompt(self, hour: int) -> str:
        """
        Construct a prompt for the AI model that focuses on the character's happiness,
        current emotional context, inventory, recent memories, and learning context at a given hour of the day.
        The AI must fully “become” the character, speaking in first-person informal English,
        check inventory privately, encounter random good/bad twist, and output exactly:
        DECISION: …
        ACTION: …
        OUTCOME: …
        HAPPINESS_CHANGE: …
        INSIGHT: …
        with no extra lines or reasoning.
        """
        time_period = self.get_time_period(hour)
        stats = self.character['current_stats']
        happiness = stats['happiness']
        loneliness = stats['loneliness']
        stress = stats['stress']
        energy = stats['energy']
        location = self.character.get('location', 'home')
        occupation = self.character.get('occupation', 'unknown job')
        goals = self.character['current_goals']
        inventory = self.character['inventory']

        # Include a random twist: 50% chance for a positive event, 50% for a negative event after decision.
        twist = random.choice(["positive", "negative"])

        prompt = (
            f"I am {self.character['name']}.\n"
            f"Day {self.character['current_day']}, Hour {hour} ({time_period}).\n"
            f"I'm at {location}, working as a {occupation}.\n"
            f"Happiness: {happiness:.1f}/10, Energy: {energy:.1f}/10, "
            f"Loneliness: {loneliness:.1f}/10, Stress: {stress:.1f}/10.\n"
            f"Core desire: {self.character['core_desire']}.\n"
            f"My current goals: {goals}.\n"
            f"My inventory: {inventory}.\n"
            f"Recent actions: {self.get_recent_context()}.\n"
            f"Learning context: {self.get_learning_context()}.\n\n"
            "I must decide what to do next, thinking as myself. First, I check my inventory and plan how to get what I lack. "
            "Then I carry out that plan, step by step. At some point, a random twist happens: "
            f"it will be a {twist} unexpected event. I deal with it as it comes. "
            "I speak in first-person informal English, like I’m texting a friend: “I’m hungry, what do I eat?”, “Nobody picks up my call”, etc. "
            "Do not reveal any private thought or chain of thought—only output the final result as if I lived it.\n\n"
            "Output exactly these five lines with no extra explanation:\n"
            "DECISION: [What I decide to do and how I handle obstacles, including inventory check]\n"
            "ACTION: [Step-by-step what I actually do, including encountering the random twist]\n"
            "OUTCOME: [How I feel at the end, what I learned, and whether I succeeded or not]\n"
            "HAPPINESS_CHANGE: [A number between -2 and +3 for net happiness change]\n"
            "INSIGHT: [One sentence about what I realize about life or happiness]\n"
        )
        return prompt

    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Query the Qwen3-4B-4bit model with the given prompt.
        If the model fails or is unavailable, fallback to a simple structured response.
        """
        if not self.model or not self.tokenizer:
            return self.generate_fallback_response(prompt)

        try:
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            print(f"⚠️ Model generation failed: {e}")
            return self.generate_fallback_response(prompt)

    def generate_fallback_response(self, prompt: str) -> str:
        """
        Simple rule-based fallback if model can't generate a response.
        Returns an informal English scenario with inventory check and a random positive/negative twist.
        """
        lower_prompt = prompt.lower()
        inventory = self.character['inventory']
        twist = random.choice(["positive", "negative"])

        # Example handling for skincare
        if "skincare" in lower_prompt or "skincare" in prompt.lower():
            if inventory.get("skincare_kit", 0) == 1:
                decision = "I see I have a skincare kit, so I’ll do my routine"
                action = (
                    "1. I wash my face with warm water.\n"
                    "2. I take my moisturizer from the kit and gently apply it.\n"
                    "3. I realize I’m out of toner, so I use some cooled green tea instead.\n"
                    "4. I pat my face dry and put on a clean shirt."
                )
                # random twist
                if twist == "negative":
                    outcome = "Halfway through, the power went out for a few minutes, so I had to wait, and I felt frustrated before finishing."
                    happiness_change = -0.2
                    insight = "Even small rituals can be ruined by unexpected annoyances"
                else:
                    outcome = "I finish my routine and feel refreshed; a text from Sarah saying “hey” lifts my mood."
                    happiness_change = +0.8
                    insight = "Simple self-care and a friend’s message can really boost my day"
            else:
                decision = "I want to do a skincare routine but I don’t have the kit, so I need to figure something else out"
                action = (
                    "1. I check my cabinet—no skincare kit.\n"
                    "2. I realize I have aloe vera gel, so I'll use that as a moisturizer.\n"
                    "3. I apply aloe vera to my face, then splash cold water.\n"
                    "4. I pat my face dry with a clean towel."
                )
                if twist == "negative":
                    outcome = "The aloe stung a bit because I had a small cut, so my face is slightly red and I feel annoyed."
                    happiness_change = -0.5
                    insight = "Sometimes improv helps, but not everything works perfectly"
                else:
                    outcome = "My face feels a bit better anyway, and I remembered an old lotion in my bag that I could use later."
                    happiness_change = +0.3
                    insight = "Adapting with what I have still counts as self-care"
        # Example for hunger
        elif "hungry" in lower_prompt or "eat" in lower_prompt or "i’m hungry" in lower_prompt:
            if "bread" in inventory.get("food_items", []):
                decision = "I’m starving, saw bread in the fridge, so I’ll make a simple sandwich"
                action = (
                    "1. I grab the bread and some cheese.\n"
                    "2. I toast the bread, but my toaster is broken.\n"
                    "3. I improvise by heating bread in a pan until it’s warm.\n"
                    "4. I add cheese and press it in the pan until it melts."
                )
                if twist == "negative":
                    outcome = "Cheese stuck to the pan, I burned my finger a bit and got annoyed, but I ate it anyway."
                    happiness_change = -0.1
                    insight = "Even a simple meal can go sideways, but at least I ate"
                else:
                    outcome = "The sandwich turned out surprisingly tasty, I feel full and content."
                    happiness_change = +1.0
                    insight = "Using what’s available can lead to pleasant surprises"
            else:
                decision = "I’m really hungry and there’s no food at home, so I need to find something to eat"
                action = (
                    "1. I look around—no food items.\n"
                    "2. I check my wallet—only $3. Tacos cost $5 nearby.\n"
                    "3. I decide to walk to a convenience store.\n"
                    "4. I find a $1 cup of instant ramen, buy it, and return home.\n"
                    "5. I cook the ramen on the stove."
                )
                if twist == "negative":
                    outcome = "The ramen packet was old and tasted weird, I felt sick afterward."
                    happiness_change = -0.8
                    insight = "Desperation meals often backfire"
                else:
                    outcome = "The ramen actually hit the spot, I feel warm and less hungry."
                    happiness_change = +0.5
                    insight = "Even simple comfort food helps in tough times"
        # Example for phone communication
        elif "call" in lower_prompt or "phone" in lower_prompt or "message" in lower_prompt:
            if inventory.get("phone_credit", 0) > 0:
                decision = "I want to call Sarah, I have enough phone credit"
                action = (
                    "1. I dial Sarah’s number.\n"
                    "2. She doesn’t pick up, so I send a text: 'Hey, just checking in.'\n"
                    "3. Her reply says she’s busy but hopes I’m okay."
                )
                if twist == "negative":
                    outcome = "After texting, I see I misspelled her name and feel embarrassed and frustrated."
                    happiness_change = -0.4
                    insight = "Small mistakes can ruin a good vibe"
                else:
                    outcome = "She texts back encouraging words, I feel supported."
                    happiness_change = +0.7
                    insight = "A friend’s quick message can lift my spirit"
            else:
                decision = "I want to call but I have no phone credit, so I need to do something else"
                action = (
                    "1. I check phone credit—it's zero.\n"
                    "2. I remember free Wi-Fi at the cafe down the street.\n"
                    "3. I grab my jacket and head out.\n"
                    "4. I log into Wi-Fi, use WhatsApp to message Sarah."
                )
                if twist == "negative":
                    outcome = "Cafe’s Wi-Fi is down, I can’t connect. I feel annoyed and lonely."
                    happiness_change = -0.7
                    insight = "Plans can fail even when you think you’ve prepared"
                else:
                    outcome = "WhatsApp goes through, and we chat for a bit. I feel less lonely."
                    happiness_change = +0.6
                    insight = "Sometimes a bit of effort can reconnect you to people"
        # Example for work
        elif "work" in lower_prompt or "job" in lower_prompt or "computer" in lower_prompt:
            decision = "I need to do some work even though I'm low on energy"
            if inventory.get("money", 0) >= 3:
                action = (
                    "1. I open my laptop but see the battery is low.\n"
                    "2. I grab $3 from my wallet, head to a cafe to charge.\n"
                    "3. I buy a cheap coffee for $2 and use the outlet.\n"
                    "4. After charging, I return and work on a draft email."
                )
                if twist == "negative":
                    outcome = "The cafe’s power flickered, my laptop died again mid-sentence. I felt frustrated."
                    happiness_change = -0.5
                    insight = "Even small efforts can be thwarted by luck"
                else:
                    outcome = "I finished the email, felt productive and relieved."
                    happiness_change = +0.8
                    insight = "A bit of effort can pay off"
            else:
                action = (
                    "1. I try to plug in my laptop at home but no outlet nearby.\n"
                    "2. I have no money to get a coffee, so I decide to take a break.\n"
                    "3. I stretch for a few minutes and take a quick walk."
                )
                if twist == "negative":
                    outcome = "It started raining during my walk, I got wet and annoyed."
                    happiness_change = -0.6
                    insight = "Sometimes breaks don’t go as planned"
                else:
                    outcome = "I feel a bit more awake after walking and decide to nap and revisit work later."
                    happiness_change = +0.3
                    insight = "Stepping away can refresh your mind"
        # Default self-care scenario
        else:
            decision = "I decide to take a short break and care for myself"
            action = (
                "1. I stand up and stretch.\n"
                "2. I open a window to let fresh air in.\n"
                "3. I drink a glass of water.\n"
                "4. I sit quietly and breathe for a few minutes."
            )
            if twist == "negative":
                outcome = "My neighbor started playing loud music, so I couldn't relax and felt irritated."
                happiness_change = -0.3
                insight = "External noise can ruin a peaceful moment"
            else:
                outcome = "I feel calmer and more centered after that short break."
                happiness_change = +0.7
                insight = "Small pauses can significantly boost mood"

        return (
            f"DECISION: {decision}\n"
            f"ACTION: {action}\n"
            f"OUTCOME: {outcome}\n"
            f"HAPPINESS_CHANGE: {happiness_change:+.1f}\n"
            f"INSIGHT: {insight}"
        )

    def parse_ai_response(self, response: str) -> dict:
        """
        Parse the AI's response text into structured data fields:
        decision, action, outcome, happiness_change (float), insight.
        """
        data = {
            'decision': '',
            'action': '',
            'outcome': '',
            'happiness_change': 0.0,
            'insight': ''
        }
        lines = response.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('DECISION:'):
                data['decision'] = stripped.replace('DECISION:', '').strip()
            elif stripped.startswith('ACTION:'):
                data['action'] = stripped.replace('ACTION:', '').strip()
            elif stripped.startswith('OUTCOME:'):
                data['outcome'] = stripped.replace('OUTCOME:', '').strip()
            elif stripped.startswith('HAPPINESS_CHANGE:'):
                try:
                    change_str = stripped.replace('HAPPINESS_CHANGE:', '').strip()
                    data['happiness_change'] = float(change_str.replace('+', ''))
                except ValueError:
                    data['happiness_change'] = random.uniform(-0.5, 1.5)
            elif stripped.startswith('INSIGHT:'):
                data['insight'] = stripped.replace('INSIGHT:', '').strip()
        return data

    def update_character_from_action(self, parsed_response: dict) -> None:
        """
        Given the parsed response from the AI, update the character's state:
        - Adjust happiness
        - Record successes and failures in learning_data
        - Append memory of the action
        - Possibly evolve personality
        """
        happiness_change = parsed_response['happiness_change']
        current_happiness = self.character['current_stats']['happiness']
        new_happiness = max(0.0, min(10.0, current_happiness + happiness_change))
        self.character['current_stats']['happiness'] = new_happiness

        strategy_str = f"{parsed_response['decision']} -> {parsed_response['outcome']}"
        if happiness_change > 1.0:
            if strategy_str not in self.learning_data['successful_strategies']:
                self.learning_data['successful_strategies'].append(strategy_str)
        elif happiness_change < -0.5:
            if strategy_str not in self.learning_data['failed_strategies']:
                self.learning_data['failed_strategies'].append(strategy_str)

        memory_entry = {
            'hour': len(self.memories['recent_actions']) % 24,
            'decision': parsed_response['decision'],
            'outcome': parsed_response['outcome'],
            'happiness_impact': happiness_change,
            'insight': parsed_response['insight'],
            'summary': f"{parsed_response['decision']} - felt {happiness_change:+.1f}"
        }
        self.memories['recent_actions'].append(memory_entry)

        if len(self.memories['recent_actions']) > 50:
            self.memories['recent_actions'] = self.memories['recent_actions'][-50:]

        if abs(happiness_change) > 1.5:
            self.evolve_personality(parsed_response['outcome'])

    def live_one_day(self) -> dict:
        """
        Simulate one full day (24 hours) of the character's life.
        For each hour:
          - Construct a deeply engineered prompt
          - Generate or fallback response
          - Parse response, update character
        At the end, create a daily summary and advance the day counter.
        Save all data to disk before returning summary.
        """
        print(f"\n🌅 DAY {self.character['current_day']} BEGINS for {self.character['name']}")
        start_happiness = self.character['current_stats']['happiness']
        print(f"😊 Starting happiness: {start_happiness:.1f}/10")

        # Possibly update goals before the day begins
        self.dynamic_goal_update()

        daily_events = []

        for hour in range(1, 25):
            print(f"\n⏰ Hour {hour}/24...")

            prompt = self.happiness_focused_prompt(hour)
            response = self.generate_response(prompt)

            # Print the full response so we see the structured scenario
            print("🤖 AI Response:")
            print(response)

            parsed = self.parse_ai_response(response)
            self.update_character_from_action(parsed)

            daily_events.append({
                'hour': hour,
                'decision': parsed['decision'],
                'happiness_change': parsed['happiness_change'],
                'insight': parsed['insight']
            })

            current_happiness = self.character['current_stats']['happiness']
            print(f"📝 DECISION: {parsed['decision']}")
            print(f"😊 Happiness: {current_happiness:.1f}/10 ({parsed['happiness_change']:+.1f})")

            time.sleep(0.5)

        end_happiness = self.character['current_stats']['happiness']
        total_change = end_happiness - start_happiness

        daily_summary = {
            'day': self.character['current_day'],
            'start_happiness': start_happiness,
            'end_happiness': end_happiness,
            'total_change': total_change,
            'events': daily_events,
            'goals_at_day_end': self.character['current_goals'].copy(),
            'personality_snapshot': self.character['personality'].copy()
        }

        self.daily_logs['logs'].append(daily_summary)

        print(f"\n🌙 DAY {self.character['current_day']} COMPLETE")
        print(f"😊 Happiness change: {total_change:+.1f}")
        print(f"🎯 Current goals: {self.character['current_goals']}")

        self.character['current_day'] += 1
        self.save_all_data()

        return daily_summary


def main():
    """
    Entry point for running the AI Life Simulation.
    Prompts for a character name (default 'Alex') and runs one simulated day.
    At the end, prints a brief summary and indicates where data is saved.
    """
    print("🌟 AI LIFE SIMULATION - DYNAMIC HAPPINESS-FOCUSED CHARACTER")
    print("=" * 60)

    character_name_input = input("Enter character name (or press Enter for 'Alex'): ").strip()
    if not character_name_input:
        character_name_input = "Alex"

    sim = AILifeSimulation(character_name_input)
    daily_summary = sim.live_one_day()

    print(f"\n📊 DAILY SUMMARY:")
    print(f"Total happiness change: {daily_summary['total_change']:+.1f}")
    significant_moments = len([e for e in daily_summary['events'] if abs(e['happiness_change']) > 1])
    print(f"Most significant moments: {significant_moments}")
    print(f"Current goals: {sim.character['current_goals']}")

    print(f"\n💾 All data saved to ai_life_data/ folder")
    print(f"🔄 Run again to continue living Day {sim.character['current_day']}!")


if __name__ == "__main__":
    main()
