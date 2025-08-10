#!/usr/bin/env python3
"""
Download and process standard instruction datasets for training.
Supports popular datasets like Alpaca, OpenAssistant, and others.
"""

import json
import requests
import os
from typing import List, Dict
import argparse


def download_alpaca_dataset():
    """Download the Stanford Alpaca dataset."""
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    
    print("Downloading Stanford Alpaca dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Downloaded {len(data)} examples from Alpaca dataset")
        return data
    else:
        raise Exception(f"Failed to download Alpaca dataset: {response.status_code}")


def download_dolly_dataset():
    """Download the Databricks Dolly dataset (subset for testing)."""
    # This is a subset - for full dataset you'd need to use HuggingFace datasets library
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/raw/main/databricks-dolly-15k.jsonl"
    
    print("Downloading Databricks Dolly dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse JSONL format
        lines = response.text.strip().split('\n')
        data = []
        for line in lines[:1000]:  # Take first 1000 examples
            if line.strip():
                item = json.loads(line)
                # Convert to our format
                formatted_item = {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("context", ""),
                    "output": item.get("response", "")
                }
                data.append(formatted_item)
        
        print(f"Downloaded {len(data)} examples from Dolly dataset")
        return data
    else:
        raise Exception(f"Failed to download Dolly dataset: {response.status_code}")


def create_alpaca_style_dataset():
    """Create a larger dataset based on Alpaca format with diverse examples."""
    # This creates a substantial dataset with varied instruction types
    base_examples = [
        # Reasoning and problem solving
        {
            "instruction": "Solve this math problem step by step.",
            "input": "If a train travels 120 km in 2 hours, what is its average speed?",
            "output": "To find average speed, I need to divide distance by time.\n\nGiven:\n- Distance = 120 km\n- Time = 2 hours\n\nFormula: Average speed = Distance ÷ Time\n\nCalculation:\nAverage speed = 120 km ÷ 2 hours = 60 km/hour\n\nTherefore, the train's average speed is 60 kilometers per hour."
        },
        {
            "instruction": "Explain the cause and effect relationship in this scenario.",
            "input": "The city installed more streetlights, and crime rates decreased by 30%.",
            "output": "This scenario demonstrates a negative correlation between lighting and crime:\n\nCause: Installation of additional streetlights\n- Increased visibility in previously dark areas\n- Enhanced surveillance capabilities\n- Greater sense of safety for residents\n\nEffect: 30% reduction in crime rates\n- Criminal activity often relies on darkness for concealment\n- Better lighting deters opportunistic crimes\n- Increased foot traffic in well-lit areas\n- Improved community engagement in public spaces\n\nThis follows the environmental crime prevention theory that improving physical conditions can reduce criminal behavior."
        },
        
        # Creative and artistic tasks
        {
            "instruction": "Write a creative story opening that hooks the reader.",
            "input": "Genre: Mystery, Setting: Old lighthouse, Character: Marine biologist",
            "output": "Dr. Elena Vasquez had studied ocean life for fifteen years, but she'd never seen anything like the bioluminescent trail leading straight to the abandoned Gull's Rest Lighthouse. The ethereal blue glow pulsed beneath the waves in a pattern too regular to be natural, too complex to be coincidence. As her boat drifted closer to the rocky shore, she realized the light was responding to her presence—and something was responding back from inside the lighthouse that had been dark for over a century."
        },
        {
            "instruction": "Describe a character's emotions through their actions, not direct statements.",
            "input": "Character: Sarah, Emotion: Nervous about job interview",
            "output": "Sarah checked her phone for the third time in two minutes, even though she knew it was only 8:47 AM. Her fingers drummed against her portfolio while she rehearsed her opening statement under her breath. She smoothed her blazer, adjusted her collar, and immediately smoothed the blazer again. The receptionist's friendly smile made her realize she'd been chewing her lip, and she quickly pressed them together, tasting the metallic hint of anxiety. Her heel tapped an erratic rhythm against the marble floor until she noticed and forced her foot to stillness."
        },
        
        # Technical explanations
        {
            "instruction": "Explain a complex technical concept in simple terms.",
            "input": "Machine learning neural networks",
            "output": "Think of a neural network like a very sophisticated pattern recognition system, similar to how your brain learns to recognize faces.\n\nImagine you're teaching a computer to recognize cats in photos:\n\n1. **Input Layer**: Like your eyes seeing the image\n2. **Hidden Layers**: Like your brain processing details (whiskers, pointed ears, fur texture)\n3. **Output Layer**: Like your brain saying \"That's a cat!\"\n\nThe network learns by:\n- Looking at thousands of labeled examples\n- Adjusting its internal connections when it makes mistakes\n- Gradually improving its accuracy\n\nJust like a child learns to recognize animals by seeing many examples, neural networks get better through practice with lots of data."
        },
        {
            "instruction": "Compare and contrast two technological approaches.",
            "input": "Cloud storage vs local storage for businesses",
            "output": "**Cloud Storage:**\nAdvantages: Scalable capacity, automatic backups, remote access, lower upfront costs, professional maintenance\nDisadvantages: Ongoing subscription costs, internet dependency, potential security concerns, less control\n\n**Local Storage:**\nAdvantages: Complete control, one-time cost, no internet required, potentially faster access, enhanced security control\nDisadvantages: Limited scalability, higher upfront investment, maintenance responsibility, risk of hardware failure\n\n**Best Choice Depends On:**\n- Company size and growth plans\n- Budget structure (upfront vs. ongoing costs)\n- Security requirements and compliance needs\n- Internet reliability and speed\n- Technical expertise available\n\nMany businesses use a hybrid approach, keeping critical data locally while using cloud for collaboration and backup."
        },
        
        # Educational content
        {
            "instruction": "Create a lesson plan outline for teaching a concept.",
            "input": "Topic: Photosynthesis for 8th grade students",
            "output": "**Photosynthesis Lesson Plan - 8th Grade (45 minutes)**\n\n**Objective:** Students will understand how plants convert sunlight into energy\n\n**Introduction (10 min):**\n- Hook: \"Why don't plants need to eat food like we do?\"\n- Show time-lapse of plant growth\n- Introduce key vocabulary\n\n**Main Content (25 min):**\n- Part 1: Ingredients needed (sunlight, water, CO2)\n- Part 2: The process (chloroplast role, chemical equation)\n- Part 3: Products (glucose, oxygen)\n- Interactive demo with aquatic plants showing oxygen bubbles\n\n**Activity (8 min):**\n- Students draw and label photosynthesis diagram\n- Pair-share explanations\n\n**Wrap-up (2 min):**\n- Quick review: \"What would happen if plants couldn't photosynthesize?\"\n- Preview next lesson on cellular respiration"
        },
        
        # Practical advice and how-to
        {
            "instruction": "Provide step-by-step guidance for a practical task.",
            "input": "How to prepare for a job interview",
            "output": "**Pre-Interview Preparation:**\n\n1. **Research the Company (2-3 days before)**\n   - Study company website, mission, recent news\n   - Understand the role requirements\n   - Research interviewer backgrounds on LinkedIn\n\n2. **Prepare Your Materials (1 day before)**\n   - Update resume and bring multiple copies\n   - Prepare a portfolio of relevant work\n   - Write down thoughtful questions to ask\n\n3. **Practice Your Responses**\n   - Rehearse common questions (strengths, weaknesses, why this company)\n   - Prepare STAR method examples (Situation, Task, Action, Result)\n   - Practice with a friend or mirror\n\n**Day of Interview:**\n- Arrive 10-15 minutes early\n- Dress professionally and appropriately\n- Bring breath mints and turn off phone\n- Maintain eye contact and firm handshake\n- Send thank-you email within 24 hours"
        },
        
        # Analysis and critical thinking
        {
            "instruction": "Analyze the pros and cons of a controversial topic.",
            "input": "Working from home vs. office work",
            "output": "**Working From Home:**\n\n*Pros:*\n- No commute time/costs\n- Flexible schedule and environment\n- Better work-life balance potential\n- Reduced office overhead for companies\n- Access to wider talent pool\n\n*Cons:*\n- Potential isolation and loneliness\n- Difficulty separating work and personal life\n- Communication challenges\n- Reduced spontaneous collaboration\n- Home distractions and ergonomic issues\n\n**Office Work:**\n\n*Pros:*\n- Face-to-face collaboration\n- Clear work-life boundaries\n- Better mentorship opportunities\n- Company culture and team bonding\n- Professional work environment\n\n*Cons:*\n- Commute time and expense\n- Fixed schedule and location\n- Office politics and distractions\n- Higher overhead costs\n- Limited flexibility\n\n**Conclusion:** The best approach likely depends on job type, individual preferences, and company culture. Many organizations are adopting hybrid models to capture benefits of both."
        }
    ]
    
    # Create variations and expand the dataset
    expanded_examples = []
    
    # Add the base examples
    expanded_examples.extend(base_examples)
    
    # Add more varied instruction types
    additional_categories = [
        # Language and communication
        {
            "instruction": "Rewrite this text in a more professional tone.",
            "input": "Hey there! So basically we're gonna need to push back the deadline cuz some stuff came up and we're behind schedule.",
            "output": "I am writing to inform you that we will need to request a deadline extension due to unexpected circumstances that have impacted our project timeline. We apologize for any inconvenience this may cause and appreciate your understanding as we work to deliver quality results."
        },
        
        # Scientific concepts
        {
            "instruction": "Explain a scientific phenomenon and its real-world applications.",
            "input": "Bernoulli's principle",
            "output": "Bernoulli's principle states that as the speed of a fluid increases, its pressure decreases. This occurs because energy in a fluid system is conserved - when kinetic energy (speed) increases, potential energy (pressure) must decrease.\n\n**Real-world applications:**\n- **Airplane wings**: Air moves faster over the curved top surface, creating lower pressure above and lift\n- **Carburetors**: Fast-moving air creates low pressure that draws fuel into the airstream\n- **Perfume atomizers**: Squeezing the bulb creates fast air flow that draws up liquid\n- **Baseball curveballs**: Spin creates different air speeds on each side, causing the ball to curve\n- **Venturi tubes**: Used in flow meters and vacuum systems\n\nThis principle is fundamental to understanding fluid dynamics in engineering and natural systems."
        },
        
        # Historical analysis
        {
            "instruction": "Analyze the historical significance of an event.",
            "input": "The invention of the printing press",
            "output": "The printing press, invented by Johannes Gutenberg around 1440, was one of history's most transformative technologies:\n\n**Immediate Impact:**\n- Made books affordable and widely available\n- Standardized spelling and grammar\n- Accelerated literacy rates across Europe\n\n**Long-term Consequences:**\n- **Religious Reform**: Enabled Protestant Reformation by spreading new ideas quickly\n- **Scientific Revolution**: Facilitated sharing of scientific discoveries and methods\n- **Democratic Ideas**: Spread of political philosophy and rights concepts\n- **Cultural Preservation**: Standardized and preserved literature and knowledge\n\n**Modern Parallels:**\nThe printing press was to the Renaissance what the internet is to the modern era - a revolutionary communication technology that democratized information access and accelerated social change.\n\n**Legacy:**\nIt laid the foundation for mass communication, public education, and the modern knowledge economy."
        }
    ]
    
    expanded_examples.extend(additional_categories)
    
    return expanded_examples


def process_dataset_for_training(data: List[Dict], output_file: str):
    """Process and save dataset in the correct format."""
    print(f"Processing {len(data)} examples...")
    
    # Ensure all examples have the required fields
    processed_data = []
    for example in data:
        processed_example = {
            "instruction": example.get("instruction", "").strip(),
            "input": example.get("input", "").strip(),
            "output": example.get("output", "").strip()
        }
        
        # Skip examples with missing critical information
        if processed_example["instruction"] and processed_example["output"]:
            processed_data.append(processed_example)
    
    print(f"Kept {len(processed_data)} valid examples")
    
    # Save the processed dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved processed dataset to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Download and process standard instruction datasets')
    parser.add_argument('--dataset', type=str, choices=['alpaca', 'dolly', 'synthetic'], 
                       default='alpaca', help='Which dataset to use')
    parser.add_argument('--output', type=str, default='standard_instruction_data.json',
                       help='Output file name')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to include')
    
    args = parser.parse_args()
    
    try:
        if args.dataset == 'alpaca':
            print("Using Stanford Alpaca dataset...")
            data = download_alpaca_dataset()
        elif args.dataset == 'dolly':
            print("Using Databricks Dolly dataset...")
            data = download_dolly_dataset()
        elif args.dataset == 'synthetic':
            print("Using high-quality synthetic dataset...")
            data = create_alpaca_style_dataset()
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Limit dataset size if requested
        if args.max_examples and len(data) > args.max_examples:
            print(f"Limiting dataset to {args.max_examples} examples")
            data = data[:args.max_examples]
        
        # Process and save
        output_file = f"/Users/michaelli/Documents/CS336_Assignment/{args.output}"
        process_dataset_for_training(data, output_file)
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Process the dataset:")
        print(f"   python create_processed_instructions.py \\")
        print(f"     --input_file {output_file} \\")
        print(f"     --output_file processed_standard_instructions.pkl \\")
        print(f"     --vocab_file /Users/michaelli/Downloads/CS336_Assignment_1/owt_bpe_vocab.pkl \\")
        print(f"     --merges_file /Users/michaelli/Downloads/CS336_Assignment_1/owt_bpe_merges.pkl")
        print()
        print("2. Train the model:")
        print("   python train.py \\")
        print("     --dataset owt \\")
        print("     --instruction_data processed_standard_instructions.pkl \\")
        print("     --max_steps 3000 \\")
        print("     --checkpoint_every 500 \\")
        print("     --learning_rate 5e-6")
        print()
        print("3. Test generalization:")
        print("   python test_generalization.py --dataset owt")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nFallback: Creating high-quality synthetic dataset...")
        data = create_alpaca_style_dataset()
        output_file = f"/Users/michaelli/Documents/CS336_Assignment/synthetic_instruction_data.json"
        process_dataset_for_training(data, output_file)


if __name__ == "__main__":
    main()
