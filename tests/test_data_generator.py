#!/usr/bin/env python3
"""
Generate test data with varying levels of AI slop for comprehensive testing.
"""

import json
import random
from pathlib import Path


def generate_clean_human_texts() -> list[tuple[str, str, float]]:
    """Generate clean, human-written texts (low slop scores expected)."""
    return [
        # Personal narratives
        (
            "personal_001",
            "I woke up this morning feeling groggy. The coffee machine was broken again, so I had to make instant coffee. It tasted terrible, but I needed the caffeine to function.",
            "personal",
            0.1,
        ),
        (
            "personal_002",
            "My grandmother's recipe for apple pie has been passed down for three generations. The secret is in the crust - she uses lard instead of butter, which makes it incredibly flaky.",
            "personal",
            0.1,
        ),
        (
            "personal_003",
            "The traffic was awful today. I sat in my car for an hour just to go three miles. I should have taken the subway, but I was running late and didn't want to deal with the crowds.",
            "personal",
            0.1,
        ),
        # Technical writing
        (
            "tech_001",
            "The bug was in the database connection pool. When we increased the pool size from 10 to 50, the timeout errors disappeared. The issue was that we had too many concurrent requests.",
            "technical",
            0.1,
        ),
        (
            "tech_002",
            "I spent three hours debugging this CSS issue. Turns out the problem was a missing semicolon on line 47. One character caused the entire stylesheet to break.",
            "technical",
            0.1,
        ),
        (
            "tech_003",
            "The server crashed at 3 AM. I had to SSH in and restart the services manually. The logs showed a memory leak in the Python application that was consuming all available RAM.",
            "technical",
            0.1,
        ),
        # Creative writing
        (
            "creative_001",
            "The old lighthouse stood sentinel on the rocky cliff, its beam cutting through the fog like a sword. Sarah could hear the waves crashing against the shore below.",
            "creative",
            0.1,
        ),
        (
            "creative_002",
            "Jake's hands trembled as he opened the letter. The paper was yellowed with age, and the handwriting was barely legible. It was from his father, written twenty years ago.",
            "creative",
            0.1,
        ),
        (
            "creative_003",
            "The garden was overgrown and wild, but beautiful in its own way. Roses climbed up the trellis, and morning glories wound their way through the fence posts.",
            "creative",
            0.1,
        ),
        # Academic writing
        (
            "academic_001",
            "The study examined the relationship between sleep quality and cognitive performance in college students. Participants completed questionnaires and underwent neuropsychological testing.",
            "academic",
            0.1,
        ),
        (
            "academic_002",
            "Climate change poses significant challenges for agricultural systems worldwide. Farmers must adapt their practices to cope with changing precipitation patterns and temperature extremes.",
            "academic",
            0.1,
        ),
        (
            "academic_003",
            "The economic impact of the pandemic has been profound and far-reaching. Unemployment rates soared, supply chains were disrupted, and consumer behavior shifted dramatically.",
            "academic",
            0.1,
        ),
        # Conversational
        (
            "conversational_001",
            "Hey, did you see that movie last night? I thought it was pretty good, but the ending was kind of weird. What did you think?",
            "conversational",
            0.1,
        ),
        (
            "conversational_002",
            "I'm so tired of this weather. It's been raining for like a week straight. I just want to see the sun again, you know?",
            "conversational",
            0.1,
        ),
        (
            "conversational_003",
            "That restaurant we went to was amazing. The pasta was incredible, and the service was perfect. We should definitely go back there sometime.",
            "conversational",
            0.1,
        ),
    ]


def generate_moderate_ai_texts() -> list[tuple[str, str, float]]:
    """Generate texts with moderate AI assistance (medium slop scores expected)."""
    return [
        # Business emails with AI polish
        (
            "business_001",
            "I hope this email finds you well. I wanted to reach out to discuss the upcoming project deadline and ensure we're all aligned on the deliverables. Please let me know if you have any questions or concerns.",
            "business",
            0.4,
        ),
        (
            "business_002",
            "Thank you for your interest in our services. We would be delighted to schedule a meeting to discuss your requirements in detail. Our team is committed to providing exceptional solutions tailored to your needs.",
            "business",
            0.4,
        ),
        (
            "business_003",
            "I wanted to follow up on our previous conversation regarding the proposal. We believe our solution offers significant value and would appreciate the opportunity to present our findings to your team.",
            "business",
            0.4,
        ),
        # Product descriptions
        (
            "product_001",
            "This innovative product combines cutting-edge technology with user-friendly design to deliver exceptional performance. Its sleek appearance and intuitive interface make it perfect for both beginners and professionals.",
            "product",
            0.4,
        ),
        (
            "product_002",
            "Experience the future of home automation with our smart device. Featuring advanced AI capabilities and seamless integration, it transforms your living space into a connected, intelligent environment.",
            "product",
            0.4,
        ),
        (
            "product_003",
            "Our premium service offers unparalleled convenience and reliability. With 24/7 customer support and state-of-the-art infrastructure, we ensure your business operations run smoothly and efficiently.",
            "product",
            0.4,
        ),
        # Educational content
        (
            "educational_001",
            "Learning a new language opens doors to countless opportunities. Our comprehensive program utilizes proven methodologies and interactive exercises to accelerate your progress and build confidence.",
            "educational",
            0.4,
        ),
        (
            "educational_002",
            "Understanding financial markets requires both theoretical knowledge and practical experience. Our course provides real-world examples and hands-on activities to develop essential skills.",
            "educational",
            0.4,
        ),
        (
            "educational_003",
            "Effective communication is crucial in today's interconnected world. Our workshop focuses on developing clear, persuasive messaging that resonates with diverse audiences across various platforms.",
            "educational",
            0.4,
        ),
        # Marketing content
        (
            "marketing_001",
            "Transform your business with our revolutionary platform. Join thousands of satisfied customers who have already experienced the benefits of our innovative approach to digital transformation.",
            "marketing",
            0.4,
        ),
        (
            "marketing_002",
            "Don't miss out on this exclusive opportunity. Limited time offer with significant savings on our premium packages. Act now to secure your spot and unlock your potential.",
            "marketing",
            0.4,
        ),
        (
            "marketing_003",
            "Discover the secret to success with our proven methodology. Our expert team has helped countless individuals achieve their goals through personalized coaching and strategic guidance.",
            "marketing",
            0.4,
        ),
    ]


def generate_high_ai_texts() -> list[tuple[str, str, float]]:
    """Generate texts with high AI content (high slop scores expected)."""
    return [
        # Generic AI responses
        (
            "ai_001",
            "I understand your concern and I'm here to help. Let me provide you with a comprehensive solution that addresses all aspects of your inquiry. This approach has been proven effective in similar situations.",
            "ai_response",
            0.7,
        ),
        (
            "ai_002",
            "Thank you for reaching out. I appreciate your patience as I work to resolve this matter. Please rest assured that I am committed to finding the best possible outcome for all parties involved.",
            "ai_response",
            0.7,
        ),
        (
            "ai_003",
            "I would be happy to assist you with this request. Based on my analysis, I recommend implementing a strategic approach that leverages our core competencies and maximizes efficiency.",
            "ai_response",
            0.7,
        ),
        # Corporate speak
        (
            "corporate_001",
            "We are excited to announce our strategic partnership that will revolutionize the industry landscape. This collaboration represents a significant milestone in our journey toward sustainable growth and innovation.",
            "corporate",
            0.7,
        ),
        (
            "corporate_002",
            "Our organization remains committed to delivering exceptional value through our comprehensive suite of solutions. We continuously strive to exceed expectations and drive meaningful change.",
            "corporate",
            0.7,
        ),
        (
            "corporate_003",
            "The implementation of our new framework will optimize operational efficiency while maintaining the highest standards of quality and customer satisfaction across all touchpoints.",
            "corporate",
            0.7,
        ),
        # Academic AI writing
        (
            "academic_ai_001",
            "The findings of this study demonstrate a statistically significant correlation between the variables under investigation. These results contribute to the existing body of literature and provide valuable insights for future research.",
            "academic_ai",
            0.7,
        ),
        (
            "academic_ai_002",
            "The methodology employed in this research follows established protocols and adheres to ethical guidelines. The data collection process was rigorous and comprehensive, ensuring the validity and reliability of our conclusions.",
            "academic_ai",
            0.7,
        ),
        (
            "academic_ai_003",
            "This analysis reveals important patterns that warrant further investigation. The implications of these findings extend beyond the immediate scope of this study and suggest promising directions for future inquiry.",
            "academic_ai",
            0.7,
        ),
        # Generic advice
        (
            "advice_001",
            "To achieve success in this endeavor, it is essential to maintain a positive mindset and remain focused on your goals. Consistent effort and determination are key factors that will contribute to your overall achievement.",
            "advice",
            0.7,
        ),
        (
            "advice_002",
            "Effective time management is crucial for maximizing productivity and achieving desired outcomes. By prioritizing tasks and eliminating distractions, you can optimize your performance and reach your objectives.",
            "advice",
            0.7,
        ),
        (
            "advice_003",
            "Building strong relationships requires genuine communication and mutual respect. Investing time in understanding others' perspectives will create meaningful connections and foster collaborative environments.",
            "advice",
            0.7,
        ),
    ]


def generate_extreme_ai_texts() -> list[tuple[str, str, float]]:
    """Generate texts with extreme AI content (very high slop scores expected)."""
    return [
        # Repetitive AI patterns
        (
            "extreme_001",
            "I understand your concern. I understand your concern. I understand your concern. Let me help you with this matter. Let me help you with this matter. Let me help you with this matter.",
            "repetitive",
            0.9,
        ),
        (
            "extreme_002",
            "Thank you for your inquiry. Thank you for your inquiry. Thank you for your inquiry. I appreciate your patience. I appreciate your patience. I appreciate your patience.",
            "repetitive",
            0.9,
        ),
        (
            "extreme_003",
            "I'm here to help. I'm here to help. I'm here to help. Please let me know. Please let me know. Please let me know. How can I assist? How can I assist? How can I assist?",
            "repetitive",
            0.9,
        ),
        # Template-heavy content
        (
            "template_001",
            "Welcome to our platform! We're excited to have you join our community of like-minded individuals who are passionate about achieving their goals and making a positive impact in their respective fields.",
            "template",
            0.9,
        ),
        (
            "template_002",
            "Our team of experts is dedicated to providing you with the highest quality service and support. We believe in the power of collaboration and are committed to helping you succeed in all your endeavors.",
            "template",
            0.9,
        ),
        (
            "template_003",
            "Thank you for choosing our services. We value your trust and are committed to delivering exceptional results that exceed your expectations. Your satisfaction is our top priority.",
            "template",
            0.9,
        ),
        # Overly formal AI
        (
            "formal_001",
            "I hereby acknowledge receipt of your correspondence and wish to express my sincere gratitude for bringing this matter to my attention. I shall endeavor to address your concerns with the utmost diligence and professionalism.",
            "formal",
            0.9,
        ),
        (
            "formal_002",
            "It is with great pleasure that I extend my warmest regards and best wishes for your continued success. May this communication serve as a testament to our ongoing commitment to excellence.",
            "formal",
            0.9,
        ),
        (
            "formal_003",
            "I respectfully submit this response in accordance with established protocols and procedures. Please be advised that all necessary measures have been implemented to ensure optimal outcomes.",
            "formal",
            0.9,
        ),
        # Generic motivational content
        (
            "motivational_001",
            "Success is not a destination, but a journey of continuous improvement and growth. Every challenge presents an opportunity to learn, adapt, and emerge stronger than before.",
            "motivational",
            0.9,
        ),
        (
            "motivational_002",
            "Believe in yourself and your ability to overcome any obstacle. With determination, perseverance, and a positive attitude, you can achieve anything you set your mind to.",
            "motivational",
            0.9,
        ),
        (
            "motivational_003",
            "The future belongs to those who dare to dream big and take bold action. Embrace change, seize opportunities, and create the life you've always imagined.",
            "motivational",
            0.9,
        ),
    ]


def generate_real_world_texts() -> list[tuple[str, str, float]]:
    """Generate real-world texts from various sources (mixed slop scores)."""
    return [
        # News headlines
        (
            "news_001",
            "Scientists discover new species of deep-sea fish that glows in the dark. The bioluminescent creature was found 2,000 meters below the surface in the Pacific Ocean.",
            "news",
            0.2,
        ),
        (
            "news_002",
            "Local bakery wins national award for best chocolate chip cookies. Owner credits secret family recipe passed down through three generations.",
            "news",
            0.2,
        ),
        (
            "news_003",
            "City council approves new bike lane project despite opposition from some residents. Construction is expected to begin next month.",
            "news",
            0.2,
        ),
        # Social media posts
        (
            "social_001",
            "Just finished reading an amazing book! The plot twists were incredible and I couldn't put it down. Highly recommend to anyone who loves mystery novels.",
            "social",
            0.3,
        ),
        (
            "social_002",
            "Coffee shop vibes today ☕ Working on my novel and people-watching. There's something inspiring about being surrounded by other creative minds.",
            "social",
            0.3,
        ),
        (
            "social_003",
            "Weekend plans: hiking, farmers market, and maybe a movie. Sometimes the simple things are the best things. #weekendvibes",
            "social",
            0.3,
        ),
        # Reviews
        (
            "review_001",
            "This restaurant has the best pasta I've ever had. The sauce is rich and flavorful, and the pasta is perfectly cooked. Service was friendly and fast. Will definitely come back!",
            "review",
            0.3,
        ),
        (
            "review_002",
            "The hotel room was clean and comfortable, but the WiFi was terrible. Location is great - walking distance to everything. Overall decent stay for the price.",
            "review",
            0.3,
        ),
        (
            "review_003",
            "Amazing concert! The band sounded incredible live and the energy was electric. Only complaint is that the venue was too crowded and hard to see the stage.",
            "review",
            0.3,
        ),
        # Technical documentation
        (
            "docs_001",
            "To install the package, run 'pip install package-name' in your terminal. Make sure you have Python 3.7 or higher installed on your system.",
            "documentation",
            0.2,
        ),
        (
            "docs_002",
            "The API endpoint returns a JSON response with the following structure: {'status': 'success', 'data': {...}, 'message': 'Operation completed'}",
            "documentation",
            0.2,
        ),
        (
            "docs_003",
            "Error 404 occurs when the requested resource is not found. Check your URL path and ensure the resource exists before making the request.",
            "documentation",
            0.2,
        ),
        # Forum posts
        (
            "forum_001",
            "Has anyone else experienced this issue? My computer keeps freezing when I try to open large files. I've tried restarting but it keeps happening.",
            "forum",
            0.3,
        ),
        (
            "forum_002",
            "Looking for recommendations for a good hiking trail near the city. Preferably something with nice views and not too crowded. Thanks in advance!",
            "forum",
            0.3,
        ),
        (
            "forum_003",
            "Just wanted to share that I finally finished my first marathon! It was tough but so worth it. The training was brutal but crossing that finish line was incredible.",
            "forum",
            0.3,
        ),
    ]


def generate_mixed_complexity_texts() -> list[tuple[str, str, float]]:
    """Generate texts with mixed complexity and AI usage patterns."""
    return [
        # Partially AI-assisted
        (
            "mixed_001",
            "I had an interesting conversation with my colleague today about machine learning. While AI can be helpful for certain tasks, I think human creativity and intuition are still irreplaceable in many areas.",
            "mixed",
            0.5,
        ),
        (
            "mixed_002",
            "The weather has been unpredictable lately. Yesterday it was sunny and warm, but today it's cold and rainy. I'm looking forward to spring when the weather becomes more consistent.",
            "mixed",
            0.5,
        ),
        (
            "mixed_003",
            "I've been learning Spanish for six months now. It's challenging but rewarding. The grammar is complex, but I enjoy practicing with native speakers whenever I get the chance.",
            "mixed",
            0.5,
        ),
        # Human-written with AI editing
        (
            "edited_001",
            "My grandmother's stories about growing up during the war were both fascinating and heartbreaking. She had such resilience and strength that I can only hope to emulate in my own life.",
            "edited",
            0.4,
        ),
        (
            "edited_002",
            "The local farmers market has the freshest produce I've ever seen. The tomatoes taste like they were picked this morning, and the bread is still warm from the oven.",
            "edited",
            0.4,
        ),
        (
            "edited_003",
            "I spent the weekend hiking in the mountains with my dog. The trail was steep and challenging, but the view from the summit made every step worth it.",
            "edited",
            0.4,
        ),
        # Hybrid content
        (
            "hybrid_001",
            "In today's rapidly evolving digital landscape, businesses must adapt to stay competitive. However, maintaining authentic human connections remains crucial for long-term success.",
            "hybrid",
            0.6,
        ),
        (
            "hybrid_002",
            "The integration of artificial intelligence into healthcare systems offers tremendous potential for improving patient outcomes. Nevertheless, ethical considerations must be carefully addressed.",
            "hybrid",
            0.6,
        ),
        (
            "hybrid_003",
            "Sustainable development requires balancing economic growth with environmental protection. This complex challenge demands innovative solutions and collaborative efforts across all sectors.",
            "hybrid",
            0.6,
        ),
    ]


def generate_test_dataset() -> list[dict[str, str]]:
    """Generate a comprehensive test dataset with 100 texts."""
    all_texts = []

    # Collect all text categories
    all_texts.extend(generate_clean_human_texts())
    all_texts.extend(generate_moderate_ai_texts())
    all_texts.extend(generate_high_ai_texts())
    all_texts.extend(generate_extreme_ai_texts())
    all_texts.extend(generate_real_world_texts())
    all_texts.extend(generate_mixed_complexity_texts())

    # Shuffle to randomize order
    random.shuffle(all_texts)

    # Convert to dictionary format
    test_data = []
    for i, (doc_id, text, domain, expected_slop) in enumerate(all_texts):
        test_data.append(
            {
                "doc_id": f"test_{i + 1:03d}_{doc_id}",
                "text": text,
                "domain": domain,
                "expected_slop_range": expected_slop,
                "category": doc_id.split("_")[0] if "_" in doc_id else "unknown",
            }
        )

    return test_data


def save_test_dataset(output_path: str = "tests/test_dataset.json") -> None:
    """Save the test dataset to a JSON file."""
    dataset = generate_test_dataset()

    # Ensure tests directory exists
    Path(output_path).parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated test dataset with {len(dataset)} texts")
    print(f"📁 Saved to: {output_path}")

    # Print summary statistics
    categories = {}
    domains = {}
    slop_ranges = {}

    for item in dataset:
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

    print("\n📊 Dataset Summary:")
    print(f"Categories: {dict(categories)}")
    print(f"Domains: {dict(domains)}")
    print(f"Expected Slop Ranges: {dict(slop_ranges)}")


if __name__ == "__main__":
    save_test_dataset()
