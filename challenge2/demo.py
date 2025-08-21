
#!/usr/bin/env python3
"""
Demo Script for AI-Automated Voice Agent Testing Platform
Demonstrates the complete workflow from basic testing to self-correction
"""

from modules.persona_generator import PersonaGenerator, TestScenario
from modules.conversation_simulator import ConversationSimulator  
from modules.automated_tester import VoiceAgentTester
from modules.self_correcting_agent import SelfCorrectingVoiceAgent

def main():
    print("🎙️ AI-Automated Voice Agent Testing Platform Demo")
    print("=" * 60)

    # Demo 1: Generate sample personas
    print("\n📋 DEMO 1: Generating Loan Defaulter Personas")
    print("-" * 50)

    generator = PersonaGenerator()
    personas = generator.generate_batch(3)

    for i, persona in enumerate(personas, 1):
        print(f"\n{i}. {persona.name}")
        print(f"   💰 Debt: ${persona.debt_amount:,.2f} ({persona.days_past_due} days overdue)")
        print(f"   👤 Traits: {', '.join(persona.personality_traits)}")
        print(f"   💬 Style: {persona.communication_style}")
        print(f"   📊 Negotiation Likelihood: {persona.negotiation_likelihood:.2f}")
        print(f"   😰 Stress Level: {persona.financial_stress_level}/10")

    # Demo 2: Run basic conversation simulation
    print("\n\n🗣️ DEMO 2: Conversation Simulation")
    print("-" * 50)

    simulator = ConversationSimulator()

    # Basic agent prompt
    basic_prompt = """
    You are a debt collection agent. Contact customers about overdue payments.
    Be professional and try to collect payment.
    """

    # Simulate a conversation
    test_persona = personas[0]
    conversation = simulator.simulate_conversation(
        persona=test_persona,
        scenario=TestScenario.PAYMENT_REMINDER,
        agent_prompt=basic_prompt,
        max_turns=6
    )

    print(f"\nConversation with {test_persona.name}:")
    print(f"Scenario: {conversation.test_scenario}")

    for i, turn in enumerate(conversation.turns, 1):
        speaker = "AGENT 🤖" if turn.speaker == "agent" else "CUSTOMER 👤"
        print(f"\n{i}. {speaker}: {turn.message}")

    print(f"\n📈 Performance Metrics:")
    metrics = conversation.metrics
    print(f"   Overall Score: {metrics.overall_score():.3f}")
    print(f"   Repetition: {metrics.repetition_score:.3f} (lower is better)")
    print(f"   Negotiation: {metrics.negotiation_effectiveness:.3f}")
    print(f"   Relevance: {metrics.response_relevance:.3f}")
    print(f"   Compliance: {metrics.compliance_score:.3f}")
    print(f"   Empathy: {metrics.empathy_score:.3f}")
    print(f"   Resolution: {'✅ Yes' if metrics.resolution_achieved else '❌ No'}")

    # Demo 3: Automated testing
    print("\n\n🧪 DEMO 3: Automated Testing Suite")
    print("-" * 50)

    tester = VoiceAgentTester()

    print("Running automated tests with basic prompt...")
    test_results = tester.run_test_suite(
        agent_prompt=basic_prompt,
        test_count=10,  # Small number for demo
        scenarios=[TestScenario.PAYMENT_REMINDER, TestScenario.FIRST_CONTACT]
    )

    print(f"\n📊 Test Results Summary:")
    metrics = test_results['aggregate_metrics']
    print(f"   Tests Run: {test_results['total_tests']}")
    print(f"   Overall Score: {metrics['overall_score']:.3f}")
    print(f"   Resolution Rate: {metrics['resolution_rate']:.3f}")
    print(f"   Compliance Score: {metrics['compliance_score']:.3f}")

    print(f"\n💡 Top Recommendations:")
    for i, rec in enumerate(test_results['recommendations'][:3], 1):
        print(f"   {i}. {rec[:100]}...")

    # Demo 4: Self-correcting agent
    print("\n\n🔄 DEMO 4: Self-Correcting Voice Agent")
    print("-" * 50)

    print("Creating self-correcting agent...")

    # Start with a very basic prompt that needs improvement
    basic_prompt_for_improvement = """
    You are a debt collector. Call people about money they owe.
    Get them to pay.
    """

    agent = SelfCorrectingVoiceAgent(
        initial_prompt=basic_prompt_for_improvement,
        target_score=0.75,
        max_iterations=2  # Limited for demo
    )

    print("Running self-improvement cycle...")
    final_report = agent.run_improvement_cycle(tests_per_iteration=8)  # Small number for demo

    print(f"\n📈 Improvement Results:")
    print(f"   Initial Score: {final_report['initial_score']:.3f}")
    print(f"   Final Score: {final_report['final_score']:.3f}")
    print(f"   Improvement: {final_report['improvement']:+.3f}")
    print(f"   Target Achieved: {'✅ Yes' if final_report['target_achieved'] else '❌ No'}")
    print(f"   Iterations Used: {final_report['iterations_used']}")

    print(f"\n📝 Final Optimized Prompt (excerpt):")
    optimized_prompt = final_report['final_prompt']
    print(f"   {optimized_prompt[:200]}...")

    # Demo 5: Performance comparison
    print("\n\n📊 DEMO 5: Performance Comparison")
    print("-" * 50)

    print("Comparing original vs optimized agent performance...")

    # Test optimized prompt
    optimized_results = tester.run_test_suite(
        agent_prompt=optimized_prompt,
        test_count=5,  # Small number for demo
        scenarios=[TestScenario.PAYMENT_REMINDER]
    )

    print("\n📈 Performance Comparison:")
    print(f"{'Metric':<25} {'Original':<10} {'Optimized':<10} {'Change':<10}")
    print("-" * 55)

    original_metrics = test_results['aggregate_metrics']
    optimized_metrics = optimized_results['aggregate_metrics']

    metrics_to_compare = [
        ('Overall Score', 'overall_score'),
        ('Repetition Score', 'repetition_score'), 
        ('Negotiation', 'negotiation_effectiveness'),
        ('Compliance', 'compliance_score'),
        ('Empathy', 'empathy_score'),
        ('Resolution Rate', 'resolution_rate')
    ]

    for display_name, metric_key in metrics_to_compare:
        orig_val = original_metrics[metric_key]
        opt_val = optimized_metrics[metric_key]
        change = opt_val - orig_val
        change_str = f"{change:+.3f}"

        print(f"{display_name:<25} {orig_val:<10.3f} {opt_val:<10.3f} {change_str:<10}")

    print("\n🎉 Demo Complete!")
    print("\nKey Takeaways:")
    print("• Automated testing can identify specific weaknesses in voice agents")
    print("• Self-correction can significantly improve agent performance")
    print("• The system provides actionable insights for manual improvements")
    print("• Compliance and empathy metrics help ensure responsible debt collection")

    print("\n🚀 Next Steps:")
    print("• Run the full web application: streamlit run voice_agent_testing_app.py")
    print("• Experiment with different agent prompts and scenarios")
    print("• Scale up testing with larger persona sets")
    print("• Integrate with your actual voice agent platform")

if __name__ == "__main__":
    main()
