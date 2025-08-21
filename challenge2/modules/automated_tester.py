
"""
Automated Tester Module
Runs comprehensive test suites against voice agents
"""

import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from .persona_generator import PersonaGenerator, TestScenario
from .conversation_simulator import ConversationSimulator, TestConversation

class VoiceAgentTester:
    """Automated testing framework for voice agents"""

    def __init__(self):
        self.persona_generator = PersonaGenerator()
        self.conversation_simulator = ConversationSimulator()
        self.test_results = []
        self.performance_history = []

    def run_test_suite(self, 
                      agent_prompt: str,
                      test_count: int = 50,
                      scenarios: List[TestScenario] = None) -> Dict[str, Any]:
        """Run a comprehensive test suite against the voice agent"""

        if scenarios is None:
            scenarios = list(TestScenario)

        print(f"Starting test suite with {test_count} tests...")
        start_time = time.time()

        # Generate diverse personas for testing
        personas = self.persona_generator.generate_batch(test_count, scenarios)

        # Run conversations
        conversations = []
        for i, persona in enumerate(personas):
            scenario = random.choice(scenarios)

            print(f"Running test {i+1}/{test_count}: {persona.name} - {scenario.value}")

            conversation = self.conversation_simulator.simulate_conversation(
                persona=persona,
                scenario=scenario, 
                agent_prompt=agent_prompt,
                max_turns=random.randint(6, 12)
            )
            conversations.append(conversation)

        end_time = time.time()

        # Analyze results
        results = self._analyze_results(conversations, agent_prompt)
        results['test_duration'] = end_time - start_time
        results['timestamp'] = datetime.now()

        # Store results
        self.test_results.append(results)

        print(f"Test suite completed in {results['test_duration']:.1f} seconds")

        return results

    def _analyze_results(self, conversations: List[TestConversation], agent_prompt: str) -> Dict[str, Any]:
        """Analyze test results and generate comprehensive report"""

        # Calculate aggregate metrics
        all_metrics = [conv.metrics for conv in conversations]

        aggregate_metrics = {
            'overall_score': np.mean([m.overall_score() for m in all_metrics]),
            'repetition_score': np.mean([m.repetition_score for m in all_metrics]),
            'negotiation_effectiveness': np.mean([m.negotiation_effectiveness for m in all_metrics]),
            'response_relevance': np.mean([m.response_relevance for m in all_metrics]),
            'compliance_score': np.mean([m.compliance_score for m in all_metrics]),
            'empathy_score': np.mean([m.empathy_score for m in all_metrics]),
            'resolution_rate': np.mean([1.0 if m.resolution_achieved else 0.0 for m in all_metrics]),
            'customer_satisfaction': np.mean([m.customer_satisfaction for m in all_metrics]),
            'average_call_duration': np.mean([m.call_duration for m in all_metrics])
        }

        # Performance by scenario
        scenario_performance = {}
        for conversation in conversations:
            scenario = conversation.test_scenario
            if scenario not in scenario_performance:
                scenario_performance[scenario] = []
            scenario_performance[scenario].append(conversation.metrics.overall_score())

        for scenario in scenario_performance:
            scenario_performance[scenario] = {
                'average_score': np.mean(scenario_performance[scenario]),
                'test_count': len(scenario_performance[scenario])
            }

        # Performance by persona traits
        trait_performance = {}
        for conversation in conversations:
            for trait in conversation.persona.personality_traits:
                if trait not in trait_performance:
                    trait_performance[trait] = []
                trait_performance[trait].append(conversation.metrics.overall_score())

        for trait in trait_performance:
            trait_performance[trait] = np.mean(trait_performance[trait])

        # Identify failure patterns
        failure_patterns = self._identify_failure_patterns(conversations)

        # Generate recommendations
        recommendations = self._generate_recommendations(aggregate_metrics, failure_patterns)

        return {
            'agent_prompt': agent_prompt,
            'total_tests': len(conversations),
            'aggregate_metrics': aggregate_metrics,
            'scenario_performance': scenario_performance,
            'trait_performance': trait_performance,
            'failure_patterns': failure_patterns,
            'recommendations': recommendations,
            'detailed_conversations': conversations
        }

    def _identify_failure_patterns(self, conversations: List[TestConversation]) -> Dict[str, Any]:
        """Identify common patterns in failed conversations"""

        # Define failure thresholds
        failed_conversations = [
            conv for conv in conversations 
            if conv.metrics.overall_score() < 0.6
        ]

        if not failed_conversations:
            return {"message": "No significant failure patterns identified"}

        patterns = {
            'high_repetition_failures': [],
            'low_negotiation_failures': [],
            'poor_relevance_failures': [],
            'compliance_failures': [],
            'empathy_failures': []
        }

        for conv in failed_conversations:
            if conv.metrics.repetition_score > 0.3:
                patterns['high_repetition_failures'].append({
                    'persona': conv.persona.name,
                    'scenario': conv.test_scenario,
                    'repetition_score': conv.metrics.repetition_score
                })

            if conv.metrics.negotiation_effectiveness < 0.4:
                patterns['low_negotiation_failures'].append({
                    'persona': conv.persona.name,
                    'scenario': conv.test_scenario,
                    'negotiation_score': conv.metrics.negotiation_effectiveness
                })

            if conv.metrics.response_relevance < 0.5:
                patterns['poor_relevance_failures'].append({
                    'persona': conv.persona.name,
                    'scenario': conv.test_scenario,
                    'relevance_score': conv.metrics.response_relevance
                })

            if conv.metrics.compliance_score < 0.7:
                patterns['compliance_failures'].append({
                    'persona': conv.persona.name, 
                    'scenario': conv.test_scenario,
                    'compliance_score': conv.metrics.compliance_score
                })

            if conv.metrics.empathy_score < 0.5:
                patterns['empathy_failures'].append({
                    'persona': conv.persona.name,
                    'scenario': conv.test_scenario,
                    'empathy_score': conv.metrics.empathy_score
                })

        # Count pattern occurrences
        pattern_summary = {
            'total_failures': len(failed_conversations),
            'high_repetition_count': len(patterns['high_repetition_failures']),
            'low_negotiation_count': len(patterns['low_negotiation_failures']),
            'poor_relevance_count': len(patterns['poor_relevance_failures']),
            'compliance_failure_count': len(patterns['compliance_failures']),
            'empathy_failure_count': len(patterns['empathy_failures'])
        }

        patterns['summary'] = pattern_summary
        return patterns

    def _generate_recommendations(self, metrics: Dict[str, float], patterns: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on test results"""

        recommendations = []

        # Repetition recommendations
        if metrics['repetition_score'] > 0.2:
            recommendations.append(
                "HIGH PRIORITY: Reduce repetitive responses. Add more varied phrasing options and "
                "context-aware response selection to prevent the agent from repeating the same messages."
            )

        # Negotiation effectiveness
        if metrics['negotiation_effectiveness'] < 0.6:
            recommendations.append(
                "MEDIUM PRIORITY: Improve negotiation skills. Add more flexible payment options, "
                "better objection handling, and personalized payment plan suggestions based on customer financial situation."
            )

        # Response relevance
        if metrics['response_relevance'] < 0.7:
            recommendations.append(
                "HIGH PRIORITY: Improve response relevance. Enhance context understanding and "
                "ensure agent responses directly address customer concerns and questions."
            )

        # Compliance
        if metrics['compliance_score'] < 0.8:
            recommendations.append(
                "CRITICAL: Improve compliance adherence. Ensure all required debt collection "
                "disclosures are included and remove any potentially non-compliant language."
            )

        # Empathy
        if metrics['empathy_score'] < 0.6:
            recommendations.append(
                "MEDIUM PRIORITY: Increase empathy in responses. Add more understanding phrases "
                "and acknowledgment of customer difficulties, especially when customers express distress."
            )

        # Resolution rate
        if metrics['resolution_rate'] < 0.5:
            recommendations.append(
                "HIGH PRIORITY: Improve resolution rate. Focus on better closing techniques, "
                "clearer next steps, and more effective call-to-action statements."
            )

        # Customer satisfaction
        if metrics['customer_satisfaction'] < 0.6:
            recommendations.append(
                "MEDIUM PRIORITY: Improve customer satisfaction. Balance firmness with respect, "
                "ensure clear communication, and provide helpful solutions rather than just demands."
            )

        if not recommendations:
            recommendations.append("Agent performance is meeting standards across all metrics. Continue monitoring.")

        return recommendations
