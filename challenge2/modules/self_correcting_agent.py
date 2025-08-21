
"""
Self-Correcting Voice Agent Module
Automatically improves voice agent prompts through iterative testing and optimization
"""

from datetime import datetime
from typing import Dict, Any, List
from .automated_tester import VoiceAgentTester

class SelfCorrectingVoiceAgent:
    """Self-improving voice agent that iteratively improves its prompts based on test results"""

    def __init__(self, initial_prompt: str, target_score: float = 0.8, max_iterations: int = 10):
        self.current_prompt = initial_prompt
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.tester = VoiceAgentTester()
        self.improvement_history = []
        self.iteration_count = 0

    def run_improvement_cycle(self, tests_per_iteration: int = 30) -> Dict[str, Any]:
        """Run the complete self-improvement cycle"""

        print(f"Starting self-improvement cycle...")
        print(f"Target score: {self.target_score}")
        print(f"Max iterations: {self.max_iterations}")

        while self.iteration_count < self.max_iterations:
            print(f"\n=== ITERATION {self.iteration_count + 1} ===")

            # Test current prompt
            results = self.tester.run_test_suite(
                agent_prompt=self.current_prompt,
                test_count=tests_per_iteration
            )

            current_score = results['aggregate_metrics']['overall_score']
            print(f"Current overall score: {current_score:.3f}")

            # Store iteration results
            iteration_data = {
                'iteration': self.iteration_count + 1,
                'prompt': self.current_prompt,
                'score': current_score,
                'results': results,
                'timestamp': datetime.now()
            }
            self.improvement_history.append(iteration_data)

            # Check if target is reached
            if current_score >= self.target_score:
                print(f"✓ Target score reached! Final score: {current_score:.3f}")
                break

            # Generate improved prompt
            new_prompt = self._improve_prompt(self.current_prompt, results)

            if new_prompt == self.current_prompt:
                print("No further improvements possible with current approach.")
                break

            self.current_prompt = new_prompt
            self.iteration_count += 1

            print(f"Generated improved prompt (iteration {self.iteration_count})")

        return self._generate_final_report()

    def _improve_prompt(self, current_prompt: str, test_results: Dict[str, Any]) -> str:
        """Generate an improved prompt based on test results"""

        recommendations = test_results['recommendations']
        metrics = test_results['aggregate_metrics']
        failure_patterns = test_results['failure_patterns']

        # Start with current prompt
        improved_sections = []

        # Add base prompt if not already comprehensive
        if "You are" not in current_prompt:
            improved_sections.append(
                "You are Sarah, a professional and empathetic debt collection specialist at Financial Recovery Services."
            )

        # Address repetition issues
        if metrics['repetition_score'] > 0.2:
            improved_sections.append("""
RESPONSE VARIETY:
- Use diverse phrasing for similar concepts
- Avoid repeating exact phrases within the same conversation
- Adapt language based on customer responses and emotional state
""")

        # Address negotiation effectiveness
        if metrics['negotiation_effectiveness'] < 0.6:
            improved_sections.append("""
NEGOTIATION STRATEGIES:
- Always offer multiple payment options (full payment, payment plan, partial payment)
- Ask about customer's financial capacity before proposing amounts
- Use collaborative language: "Let's work together to find a solution"
- Acknowledge customer's efforts and circumstances
- Be flexible with payment dates and amounts based on customer situation
""")

        # Address response relevance
        if metrics['response_relevance'] < 0.7:
            improved_sections.append("""
RESPONSE GUIDELINES:
- Listen carefully to customer concerns and address them directly
- Ask clarifying questions when customer statements are unclear
- Acknowledge customer emotions before moving to business matters
- Provide specific information when customers ask questions
- Stay on topic and avoid generic responses
""")

        # Address compliance issues
        if metrics['compliance_score'] < 0.8:
            improved_sections.append("""
COMPLIANCE REQUIREMENTS (CRITICAL):
- Always state: "This call may be recorded for quality assurance"
- Include: "This is an attempt to collect a debt. Any information obtained will be used for that purpose"
- Never use threatening language or mention legal consequences you cannot enforce
- Respect customer rights to dispute or request validation
- Follow FDCPA guidelines strictly
""")

        # Address empathy issues
        if metrics['empathy_score'] < 0.6:
            improved_sections.append("""
EMPATHY AND COMMUNICATION:
- Use understanding phrases: "I understand this is difficult"
- Acknowledge customer stress and financial challenges
- Thank customers for their time and honesty
- Express willingness to help find solutions
- Maintain respectful tone even if customer becomes upset
- Show patience with confused or anxious customers
""")

        # Address resolution rate
        if metrics['resolution_rate'] < 0.5:
            improved_sections.append("""
CALL RESOLUTION:
- Clearly summarize any agreements reached
- Provide specific next steps and timelines
- Confirm customer contact information
- Give customers a reference number or confirmation
- End with clear expectations for follow-up
""")

        # Combine sections
        if improved_sections:
            new_prompt = current_prompt + "\n\nIMPROVEMENTS BASED ON TESTING:\n" + "\n".join(improved_sections)
        else:
            # Minor refinements if no major issues
            new_prompt = current_prompt + "\n\nREFINEMENT: Focus on maintaining consistent quality across all customer interactions."

        return new_prompt

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final improvement report"""

        if not self.improvement_history:
            return {"error": "No improvement data available"}

        initial_score = self.improvement_history[0]['score']
        final_score = self.improvement_history[-1]['score']
        improvement = final_score - initial_score

        score_progression = [iteration['score'] for iteration in self.improvement_history]

        return {
            'initial_score': initial_score,
            'final_score': final_score,
            'improvement': improvement,
            'target_achieved': final_score >= self.target_score,
            'iterations_used': len(self.improvement_history),
            'score_progression': score_progression,
            'final_prompt': self.current_prompt,
            'improvement_history': self.improvement_history
        }
