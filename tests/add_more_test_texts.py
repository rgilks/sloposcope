#!/usr/bin/env python3
"""
Add more test texts to reach 100 total samples.
"""

import json
import random


def add_additional_texts():
    """Add 25 more texts to reach 100 total."""

    # Load existing dataset
    with open("tests/test_dataset.json", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Additional texts to reach 100
    additional_texts = [
        # More personal narratives
        {
            "doc_id": "personal_004",
            "text": "I forgot my umbrella today and got completely soaked walking to work. The rain came out of nowhere, and I had to sit in wet clothes all morning. Lesson learned: always check the weather forecast.",
            "domain": "personal",
            "expected_slop_range": 0.1,
            "category": "personal",
        },
        {
            "doc_id": "personal_005",
            "text": "My cat knocked over my coffee cup this morning, spilling coffee all over my laptop. Thankfully it still works, but now my keyboard is sticky and smells like coffee.",
            "domain": "personal",
            "expected_slop_range": 0.1,
            "category": "personal",
        },
        {
            "doc_id": "personal_006",
            "text": "I tried cooking a new recipe last night and it was a complete disaster. The chicken was burnt on the outside and raw on the inside. I ended up ordering pizza instead.",
            "domain": "personal",
            "expected_slop_range": 0.1,
            "category": "personal",
        },
        # More technical content
        {
            "doc_id": "tech_004",
            "text": "The database query was taking forever because we forgot to add an index on the user_id column. After adding the index, the query time dropped from 30 seconds to 0.1 seconds.",
            "domain": "technical",
            "expected_slop_range": 0.1,
            "category": "tech",
        },
        {
            "doc_id": "tech_005",
            "text": "I spent all day debugging a memory leak in our Python application. Turns out we weren't properly closing database connections in our async functions.",
            "domain": "technical",
            "expected_slop_range": 0.1,
            "category": "tech",
        },
        {
            "doc_id": "tech_006",
            "text": "The API was returning 500 errors because our Redis cache was full. We had to increase the memory limit and implement a better eviction policy.",
            "domain": "technical",
            "expected_slop_range": 0.1,
            "category": "tech",
        },
        # More creative writing
        {
            "doc_id": "creative_004",
            "text": "The old bookstore smelled of dust and paper, with sunlight streaming through the dusty windows. Sarah found the book she'd been searching for on the top shelf.",
            "domain": "creative",
            "expected_slop_range": 0.1,
            "category": "creative",
        },
        {
            "doc_id": "creative_005",
            "text": "The storm raged outside, but inside the cabin it was warm and cozy. Jake sat by the fireplace reading his grandfather's journal.",
            "domain": "creative",
            "expected_slop_range": 0.1,
            "category": "creative",
        },
        {
            "doc_id": "creative_006",
            "text": "The garden was alive with color in the morning light. Bees buzzed among the flowers, and butterflies danced from bloom to bloom.",
            "domain": "creative",
            "expected_slop_range": 0.1,
            "category": "creative",
        },
        # More business content
        {
            "doc_id": "business_004",
            "text": "We need to schedule a meeting to discuss the Q4 budget allocation. Please review the attached spreadsheet and come prepared with your department's requirements.",
            "domain": "business",
            "expected_slop_range": 0.4,
            "category": "business",
        },
        {
            "doc_id": "business_005",
            "text": "The client feedback has been overwhelmingly positive. We should leverage this success to expand into new markets and increase our market share.",
            "domain": "business",
            "expected_slop_range": 0.4,
            "category": "business",
        },
        {
            "doc_id": "business_006",
            "text": "Our quarterly results exceeded expectations across all key metrics. This performance demonstrates the effectiveness of our strategic initiatives.",
            "domain": "business",
            "expected_slop_range": 0.4,
            "category": "business",
        },
        # More AI-generated content
        {
            "doc_id": "ai_004",
            "text": "I understand your concern and I'm committed to resolving this matter promptly. Please rest assured that I will investigate this issue thoroughly and provide you with a comprehensive solution.",
            "domain": "ai_response",
            "expected_slop_range": 0.7,
            "category": "ai",
        },
        {
            "doc_id": "ai_005",
            "text": "Thank you for bringing this to my attention. I appreciate your patience as I work to address your concerns. I will ensure that all necessary steps are taken to resolve this situation.",
            "domain": "ai_response",
            "expected_slop_range": 0.7,
            "category": "ai",
        },
        {
            "doc_id": "ai_006",
            "text": "I would be happy to assist you with this request. Based on my analysis, I recommend implementing a systematic approach that addresses all aspects of your inquiry.",
            "domain": "ai_response",
            "expected_slop_range": 0.7,
            "category": "ai",
        },
        # More corporate speak
        {
            "doc_id": "corporate_004",
            "text": "Our organization is committed to fostering innovation and driving sustainable growth through strategic partnerships and operational excellence.",
            "domain": "corporate",
            "expected_slop_range": 0.7,
            "category": "corporate",
        },
        {
            "doc_id": "corporate_005",
            "text": "We are pleased to announce our continued commitment to delivering exceptional value to our stakeholders through our comprehensive suite of solutions.",
            "domain": "corporate",
            "expected_slop_range": 0.7,
            "category": "corporate",
        },
        {
            "doc_id": "corporate_006",
            "text": "The implementation of our new framework will optimize efficiency while maintaining the highest standards of quality and customer satisfaction.",
            "domain": "corporate",
            "expected_slop_range": 0.7,
            "category": "corporate",
        },
        # More real-world content
        {
            "doc_id": "news_004",
            "text": "Local library receives $50,000 grant to expand digital resources. The funding will be used to purchase new computers and online databases for community use.",
            "domain": "news",
            "expected_slop_range": 0.2,
            "category": "news",
        },
        {
            "doc_id": "news_005",
            "text": "City announces new recycling program starting next month. Residents will receive new bins and collection schedules in the mail.",
            "domain": "news",
            "expected_slop_range": 0.2,
            "category": "news",
        },
        {
            "doc_id": "news_006",
            "text": "School district hires 15 new teachers for the upcoming school year. The new hires will help reduce class sizes and improve student outcomes.",
            "domain": "news",
            "expected_slop_range": 0.2,
            "category": "news",
        },
        # More social media content
        {
            "doc_id": "social_004",
            "text": "Just discovered this amazing podcast! The host has such great insights and the production quality is incredible. Already binge-listened to 5 episodes.",
            "domain": "social",
            "expected_slop_range": 0.3,
            "category": "social",
        },
        {
            "doc_id": "social_005",
            "text": "Weekend vibes: farmers market, coffee shop, and a long walk in the park. Sometimes the simple things really are the best things.",
            "domain": "social",
            "expected_slop_range": 0.3,
            "category": "social",
        },
        {
            "doc_id": "social_006",
            "text": "My plants are thriving! 🌱 Started with just one succulent and now I have a whole collection. There's something so satisfying about watching them grow.",
            "domain": "social",
            "expected_slop_range": 0.3,
            "category": "social",
        },
        # More reviews
        {
            "doc_id": "review_004",
            "text": "This book was absolutely incredible! The character development was amazing and the plot twists kept me guessing until the very end. Couldn't put it down.",
            "domain": "review",
            "expected_slop_range": 0.3,
            "category": "review",
        },
        {
            "doc_id": "review_005",
            "text": "The service at this restaurant was terrible. We waited 45 minutes for our food and when it finally arrived, it was cold and overcooked. Won't be coming back.",
            "domain": "review",
            "expected_slop_range": 0.3,
            "category": "review",
        },
        {
            "doc_id": "review_006",
            "text": "Great product overall, but the shipping was slower than expected. The item arrived in good condition though, and the quality is exactly as described.",
            "domain": "review",
            "expected_slop_range": 0.3,
            "category": "review",
        },
    ]

    # Add the additional texts
    for text_data in additional_texts:
        # Generate unique doc_id
        text_data["doc_id"] = f"test_{len(existing_data) + 1:03d}_{text_data['doc_id']}"
        existing_data.append(text_data)

    # Shuffle the entire dataset
    random.shuffle(existing_data)

    # Save updated dataset
    with open("tests/test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Added {len(additional_texts)} more texts")
    print(f"📊 Total texts: {len(existing_data)}")

    # Print summary
    categories = {}
    domains = {}
    slop_ranges = {}

    for item in existing_data:
        cat = item["category"]
        domain = item["domain"]
        slop = item["expected_slop_range"]

        categories[cat] = categories.get(cat, 0) + 1
        domains[domain] = domains.get(domain, 0) + 1

        if slop < 0.3:
            slop_range = "Low (0.0-0.3)"
        elif slop < 0.6:
            slop_range = "Medium (0.3-0.6)"
        else:
            slop_range = "High (0.6-1.0)"

        slop_ranges[slop_range] = slop_ranges.get(slop_range, 0) + 1

    print("\n📊 Updated Dataset Summary:")
    print(f"Categories: {len(categories)} unique categories")
    print(f"Domains: {len(domains)} unique domains")
    print(f"Expected Slop Ranges: {dict(slop_ranges)}")


if __name__ == "__main__":
    add_additional_texts()
