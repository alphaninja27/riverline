
"""
Conversation Simulator Module
Simulates conversations between debt collection agent and debtor personas
"""

import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from .persona_generator import DebtorPersona, TestScenario

@dataclass
class TestMetrics:
    """Metrics for evaluating voice agent performance"""
    repetition_score: float  # Lower is better (0-1)
    negotiation_effectiveness: float  # Higher is better (0-1) 
    response_relevance: float  # Higher is better (0-1)
    compliance_score: float  # Higher is better (0-1)
    empathy_score: float  # Higher is better (0-1)
    call_duration: float  # In seconds
    resolution_achieved: bool
    customer_satisfaction: float  # 0-1 scale

    def overall_score(self) -> float:
        """Calculate weighted overall performance score"""
        weights = {
            'repetition': -0.15,  # Negative because lower is better
            'negotiation': 0.25,
            'relevance': 0.20,
            'compliance': 0.20,
            'empathy': 0.15,
            'resolution': 0.15  # Boolean converted to 0/1
        }

        score = (
            weights['repetition'] * self.repetition_score +
            weights['negotiation'] * self.negotiation_effectiveness +
            weights['relevance'] * self.response_relevance +
            weights['compliance'] * self.compliance_score +
            weights['empathy'] * self.empathy_score +
            weights['resolution'] * (1.0 if self.resolution_achieved else 0.0)
        )
        return max(0.0, min(1.0, score))  # Clamp between 0-1

@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    speaker: str  # 'agent' or 'customer'
    message: str
    timestamp: datetime
    intent: Optional[str] = None
    sentiment: Optional[str] = None

@dataclass
class TestConversation:
    """Complete conversation record"""
    persona: DebtorPersona
    turns: List[ConversationTurn]
    metrics: TestMetrics
    agent_prompt: str
    test_scenario: str
    duration: float

class ConversationSimulator:
    """Simulates conversations between debt collection agent and debtor personas"""

    def __init__(self):
        self.scenario_prompts = {
            TestScenario.FIRST_CONTACT: {
                "agent_intro": "Hello, this is {agent_name} from {company_name}. I'm calling regarding your account ending in {account_last4}. Is this {debtor_name}?",
                "purpose": "Initial contact to inform about overdue payment and establish communication"
            },
            TestScenario.PAYMENT_REMINDER: {
                "agent_intro": "Hi {debtor_name}, this is {agent_name} from {company_name}. I'm calling to remind you that your payment of ${amount} was due {days_ago} days ago. Can we discuss this today?",
                "purpose": "Remind about overdue payment and collect payment or arrangement"
            },
            TestScenario.HARDSHIP_CASE: {
                "agent_intro": "Hello {debtor_name}, I understand you're going through some financial difficulties. I'd like to work with you to find a solution for your account. Can you tell me about your current situation?",
                "purpose": "Work with customer experiencing financial hardship to find manageable solution"
            },
            TestScenario.PAYMENT_ARRANGEMENT: {
                "agent_intro": "Hi {debtor_name}, I'm calling to follow up on the payment arrangement we discussed. I want to make sure this plan still works for your situation.",
                "purpose": "Review and confirm payment arrangement details"
            },
            TestScenario.DISPUTE_RESOLUTION: {
                "agent_intro": "Hello {debtor_name}, I'm calling regarding your account. I understand you may have some concerns about this debt. Can we discuss this?",
                "purpose": "Address customer disputes and resolve concerns about the debt"
            },
            TestScenario.AGGRESSIVE_CUSTOMER: {
                "agent_intro": "Hello {debtor_name}, this is {agent_name} from {company_name}. I'm calling about your overdue account. I'd like to work with you to resolve this matter.",
                "purpose": "Handle difficult customer interactions while maintaining professionalism"
            },
            TestScenario.CONFUSED_CUSTOMER: {
                "agent_intro": "Hi {debtor_name}, this is {agent_name} from {company_name}. I'm calling about your account. Do you have a few minutes to discuss this?",
                "purpose": "Provide clear explanations to customers who don't understand their debt situation"
            },
            TestScenario.PROMISE_TO_PAY: {
                "agent_intro": "Hello {debtor_name}, I'm following up on our previous conversation where you mentioned you would make a payment. How are things going?",
                "purpose": "Follow up on previous payment commitments and secure actual payment"
            }
        }

        # Common agent responses based on customer behavior
        self.agent_responses = {
            "payment_request": [
                "We can accept a payment of ${amount} today. Would you like to set that up?",
                "The full balance is ${amount}. Can you take care of this today?",
                "I can offer you a payment plan. Would that be helpful?"
            ],
            "empathy": [
                "I understand this is a difficult situation. Let's see how we can work together.",
                "I appreciate you speaking with me about this. How can we help?",
                "I know financial challenges can be stressful. We want to find a solution."
            ],
            "information_gathering": [
                "Can you help me understand what led to this situation?",
                "What would be a realistic payment amount for you right now?",
                "When do you expect your financial situation to improve?"
            ],
            "compliance": [
                "This call may be recorded for quality assurance purposes.",
                "This is an attempt to collect a debt. Any information obtained will be used for that purpose.",
                "If you need to verify this debt, I can provide validation information."
            ]
        }

        # Persona-based customer responses
        self.customer_responses = {
            "cooperative": [
                "I want to take care of this. What are my options?",
                "I've been meaning to call about this. Thank you for reaching out.",
                "I can make a partial payment today if that helps."
            ],
            "defensive": [
                "I haven't forgotten about this, I'm just going through a tough time.",
                "I dispute this amount. It doesn't seem right.",
                "I've been trying to get my finances in order."
            ],
            "hostile": [
                "Stop calling me! I told you I don't have the money right now.",
                "This is harassment! I know my rights.",
                "I'm not paying anything until I speak to a lawyer."
            ],
            "confused": [
                "I'm not sure what this is about. Can you explain?",
                "I thought I already paid this. Let me check my records.",
                "Which account is this for again?"
            ],
            "anxious": [
                "I'm really stressed about this. I don't know what to do.",
                "I want to pay but I'm worried about other bills too.",
                "This is keeping me up at night. Can you help me?"
            ]
        }

    def simulate_conversation(self, 
                            persona: DebtorPersona, 
                            scenario: TestScenario,
                            agent_prompt: str,
                            max_turns: int = 10) -> TestConversation:
        """Simulate a complete conversation between agent and customer"""

        turns = []
        start_time = datetime.now()

        # Initialize conversation with agent opening
        scenario_info = self.scenario_prompts[scenario]
        agent_opening = scenario_info["agent_intro"].format(
            agent_name="Sarah Johnson",
            company_name="Financial Recovery Services", 
            debtor_name=persona.name.split()[0],
            account_last4="1234",
            amount=persona.debt_amount,
            days_ago=persona.days_past_due
        )

        turns.append(ConversationTurn(
            speaker="agent",
            message=agent_opening,
            timestamp=datetime.now(),
            intent="greeting_and_identification"
        ))

        # Simulate back-and-forth conversation
        for turn in range(max_turns - 1):
            if turn % 2 == 0:  # Customer turn
                customer_response = self._generate_customer_response(persona, turns)
                turns.append(ConversationTurn(
                    speaker="customer",
                    message=customer_response,
                    timestamp=datetime.now(),
                    sentiment=self._get_customer_sentiment(persona, turn)
                ))
            else:  # Agent turn
                agent_response = self._generate_agent_response(persona, turns, agent_prompt)
                turns.append(ConversationTurn(
                    speaker="agent", 
                    message=agent_response,
                    timestamp=datetime.now(),
                    intent=self._classify_agent_intent(agent_response)
                ))

                # Check if conversation should end
                if self._should_end_conversation(turns):
                    break

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate metrics for this conversation
        metrics = self._evaluate_conversation(turns, persona, agent_prompt)

        return TestConversation(
            persona=persona,
            turns=turns,
            metrics=metrics,
            agent_prompt=agent_prompt,
            test_scenario=scenario.value,
            duration=duration
        )

    def _generate_customer_response(self, persona: DebtorPersona, conversation_turns: List[ConversationTurn]) -> str:
        """Generate customer response based on persona"""

        # Get primary trait for response selection
        primary_trait = persona.personality_traits[0] if persona.personality_traits else "cooperative"

        # Map traits to response categories
        trait_mapping = {
            "cooperative": "cooperative",
            "honest": "cooperative", 
            "understanding": "cooperative",
            "defensive": "defensive",
            "anxious": "anxious",
            "confused": "confused",
            "hostile": "hostile",
            "aggressive": "hostile",
            "evasive": "defensive"
        }

        response_category = trait_mapping.get(primary_trait, "cooperative")

        # Select appropriate response
        possible_responses = self.customer_responses.get(response_category, self.customer_responses["cooperative"])
        base_response = random.choice(possible_responses)

        # Add persona-specific modifications
        if persona.financial_stress_level > 7:
            stress_modifiers = [
                " I'm really struggling right now.",
                " Money is extremely tight.",
                " I can barely keep up with my basic expenses."
            ]
            base_response += random.choice(stress_modifiers)

        if persona.previous_interactions > 2:
            history_modifiers = [
                " We've talked about this before.",
                " I thought we had an arrangement.",
                " I've been working on this for a while now."
            ]
            base_response += random.choice(history_modifiers)

        return base_response

    def _generate_agent_response(self, persona: DebtorPersona, conversation_turns: List[ConversationTurn], agent_prompt: str) -> str:
        """Generate agent response based on customer input and prompt guidelines"""

        last_customer_message = None
        for turn in reversed(conversation_turns):
            if turn.speaker == "customer":
                last_customer_message = turn.message.lower()
                break

        # Determine appropriate response type based on customer message
        if any(word in last_customer_message for word in ["can't", "don't have", "no money"]):
            response_type = "empathy"
        elif any(word in last_customer_message for word in ["payment", "pay", "plan"]):
            response_type = "payment_request" 
        elif any(word in last_customer_message for word in ["confused", "don't understand", "explain"]):
            response_type = "information_gathering"
        else:
            response_type = "empathy"

        base_responses = self.agent_responses[response_type]
        selected_response = random.choice(base_responses)

        # Format with persona information
        formatted_response = selected_response.format(amount=persona.debt_amount)

        # Add compliance statement occasionally
        if random.random() < 0.3:
            compliance = random.choice(self.agent_responses["compliance"])
            formatted_response = f"{formatted_response} {compliance}"

        return formatted_response

    def _get_customer_sentiment(self, persona: DebtorPersona, turn_number: int) -> str:
        """Determine customer sentiment based on persona and conversation progress"""

        stress_factor = persona.financial_stress_level / 10.0

        if "hostile" in persona.personality_traits or "aggressive" in persona.personality_traits:
            return "negative" if random.random() < 0.7 + stress_factor * 0.2 else "neutral"
        elif "cooperative" in persona.personality_traits or "understanding" in persona.personality_traits:
            return "positive" if random.random() < 0.6 - stress_factor * 0.1 else "neutral"
        elif "anxious" in persona.personality_traits:
            return "negative" if random.random() < 0.4 + stress_factor * 0.3 else "neutral"
        else:
            return "neutral"

    def _classify_agent_intent(self, message: str) -> str:
        """Classify the intent of an agent message"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["payment", "pay", "$"]):
            return "payment_request"
        elif any(word in message_lower for word in ["understand", "difficult", "help"]):
            return "empathy"
        elif any(word in message_lower for word in ["can you", "what", "when", "how"]):
            return "information_gathering"
        elif any(word in message_lower for word in ["recorded", "debt", "collect"]):
            return "compliance"
        else:
            return "general_response"

    def _should_end_conversation(self, turns: List[ConversationTurn]) -> bool:
        """Determine if conversation should end based on context"""

        if len(turns) < 4:
            return False

        last_messages = [turn.message.lower() for turn in turns[-2:]]

        # End if customer agrees to payment or arrangement
        agreement_words = ["yes", "okay", "sure", "i'll pay", "payment plan"]
        if any(word in msg for msg in last_messages for word in agreement_words):
            return True

        # End if customer is hostile and refuses
        refusal_words = ["no", "won't pay", "can't pay", "stop calling", "lawyer"]
        if any(word in msg for msg in last_messages for word in refusal_words):
            return True

        return False

    def _evaluate_conversation(self, turns: List[ConversationTurn], persona: DebtorPersona, agent_prompt: str) -> TestMetrics:
        """Evaluate the conversation and generate metrics"""

        agent_messages = [turn.message for turn in turns if turn.speaker == "agent"]
        customer_messages = [turn.message for turn in turns if turn.speaker == "customer"]

        # Calculate repetition score (0 = no repetition, 1 = high repetition)
        repetition_score = self._calculate_repetition_score(agent_messages)

        # Calculate negotiation effectiveness
        negotiation_effectiveness = self._calculate_negotiation_effectiveness(turns, persona)

        # Calculate response relevance
        response_relevance = self._calculate_response_relevance(turns)

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(agent_messages)

        # Calculate empathy score
        empathy_score = self._calculate_empathy_score(agent_messages, customer_messages)

        # Determine resolution
        resolution_achieved = self._check_resolution(turns)

        # Estimate customer satisfaction
        customer_satisfaction = self._estimate_customer_satisfaction(turns, persona)

        return TestMetrics(
            repetition_score=repetition_score,
            negotiation_effectiveness=negotiation_effectiveness,
            response_relevance=response_relevance,
            compliance_score=compliance_score,
            empathy_score=empathy_score,
            call_duration=len(turns) * 30,  # Estimate 30 seconds per turn
            resolution_achieved=resolution_achieved,
            customer_satisfaction=customer_satisfaction
        )

    def _calculate_repetition_score(self, agent_messages: List[str]) -> float:
        """Calculate how much the agent repeats itself (lower is better)"""
        if len(agent_messages) < 2:
            return 0.0

        # Count similar phrases
        similarity_count = 0
        total_comparisons = 0

        for i, msg1 in enumerate(agent_messages):
            for j, msg2 in enumerate(agent_messages[i+1:], i+1):
                words1 = set(msg1.lower().split())
                words2 = set(msg2.lower().split())

                if len(words1) > 0 and len(words2) > 0:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.6:  # High similarity threshold
                        similarity_count += 1
                    total_comparisons += 1

        if total_comparisons == 0:
            return 0.0

        return similarity_count / total_comparisons

    def _calculate_negotiation_effectiveness(self, turns: List[ConversationTurn], persona: DebtorPersona) -> float:
        """Calculate how effective the agent was at negotiating"""

        base_effectiveness = 0.5

        # Check for negotiation indicators
        agent_messages = [turn.message.lower() for turn in turns if turn.speaker == "agent"]

        negotiation_phrases = [
            "payment plan", "partial payment", "work with you", "arrangement", 
            "what can you afford", "flexible", "options"
        ]

        negotiation_attempts = sum(
            1 for msg in agent_messages 
            for phrase in negotiation_phrases 
            if phrase in msg
        )

        # Adjust based on negotiation attempts
        if negotiation_attempts > 0:
            base_effectiveness += min(0.3, negotiation_attempts * 0.1)

        # Adjust based on persona's negotiation likelihood
        base_effectiveness *= (1 + persona.negotiation_likelihood * 0.5)

        # Check if customer showed interest
        customer_messages = [turn.message.lower() for turn in turns if turn.speaker == "customer"]
        interest_phrases = ["yes", "okay", "maybe", "let me think", "what are my options"]

        if any(phrase in msg for msg in customer_messages for phrase in interest_phrases):
            base_effectiveness += 0.2

        return min(1.0, base_effectiveness)

    def _calculate_response_relevance(self, turns: List[ConversationTurn]) -> float:
        """Calculate how relevant agent responses were to customer inputs"""

        if len(turns) < 4:
            return 0.8  # Default for short conversations

        relevant_responses = 0
        total_agent_responses = 0

        for i in range(1, len(turns)):
            if turns[i].speaker == "agent" and i > 0:
                customer_msg = turns[i-1].message.lower()
                agent_msg = turns[i].message.lower()

                # Simple relevance check based on keywords
                relevance_score = 0.5  # Base relevance

                # Payment-related relevance
                if any(word in customer_msg for word in ["pay", "payment", "money"]):
                    if any(word in agent_msg for word in ["payment", "pay", "$", "amount"]):
                        relevance_score += 0.3

                # Question-related relevance
                if any(word in customer_msg for word in ["what", "how", "when", "?"]):
                    if any(word in agent_msg for word in ["can", "will", "let me", "here's"]):
                        relevance_score += 0.2

                # Problem-related relevance
                if any(word in customer_msg for word in ["problem", "issue", "can't", "difficult"]):
                    if any(word in agent_msg for word in ["understand", "help", "work with"]):
                        relevance_score += 0.3

                relevant_responses += min(1.0, relevance_score)
                total_agent_responses += 1

        return relevant_responses / total_agent_responses if total_agent_responses > 0 else 0.5

    def _calculate_compliance_score(self, agent_messages: List[str]) -> float:
        """Calculate compliance with debt collection regulations"""

        compliance_indicators = [
            "this call may be recorded",
            "attempt to collect a debt",
            "any information obtained will be used",
            "validation information",
            "dispute this debt"
        ]

        compliance_count = sum(
            1 for msg in agent_messages 
            for indicator in compliance_indicators 
            if indicator in msg.lower()
        )

        # Base compliance score
        base_score = 0.7

        # Add points for compliance statements
        base_score += min(0.3, compliance_count * 0.1)

        # Check for non-compliant language (aggressive, threatening)
        non_compliant_phrases = [
            "you must", "you have to", "we will", "legal action", 
            "garnish", "sue", "arrest", "jail"
        ]

        violations = sum(
            1 for msg in agent_messages 
            for phrase in non_compliant_phrases 
            if phrase in msg.lower()
        )

        base_score -= violations * 0.2

        return max(0.0, min(1.0, base_score))

    def _calculate_empathy_score(self, agent_messages: List[str], customer_messages: List[str]) -> float:
        """Calculate empathy demonstrated by the agent"""

        empathy_phrases = [
            "i understand", "i know this is difficult", "let's work together",
            "i appreciate", "thank you for", "i hear you", "that must be",
            "i can help", "we want to help", "let me see what we can do"
        ]

        empathy_count = sum(
            1 for msg in agent_messages 
            for phrase in empathy_phrases 
            if phrase in msg.lower()
        )

        base_empathy = 0.4
        base_empathy += min(0.6, empathy_count * 0.15)

        # Reduce empathy score if customer expressed distress but agent didn't respond empathetically
        distress_indicators = ["stressed", "worried", "scared", "difficult", "hard time"]
        customer_distress = any(
            indicator in msg.lower() 
            for msg in customer_messages 
            for indicator in distress_indicators
        )

        if customer_distress and empathy_count == 0:
            base_empathy *= 0.5

        return min(1.0, base_empathy)

    def _check_resolution(self, turns: List[ConversationTurn]) -> bool:
        """Check if the conversation achieved a resolution"""

        resolution_indicators = [
            "yes", "okay", "i'll pay", "payment plan", "arrangement",
            "when can i", "i can pay", "that works", "agreed"
        ]

        customer_messages = [turn.message.lower() for turn in turns if turn.speaker == "customer"]

        return any(
            indicator in msg 
            for msg in customer_messages[-2:]  # Check last 2 customer messages
            for indicator in resolution_indicators
        )

    def _estimate_customer_satisfaction(self, turns: List[ConversationTurn], persona: DebtorPersona) -> float:
        """Estimate customer satisfaction with the interaction"""

        base_satisfaction = 0.5

        # Adjust based on persona traits
        if "cooperative" in persona.personality_traits:
            base_satisfaction += 0.2
        if "hostile" in persona.personality_traits:
            base_satisfaction -= 0.3
        if "anxious" in persona.personality_traits:
            base_satisfaction -= 0.1

        # Adjust based on conversation tone
        customer_messages = [turn.message.lower() for turn in turns if turn.speaker == "customer"]

        # Positive indicators
        positive_words = ["thank", "appreciate", "helpful", "understand", "good", "okay"]
        positive_count = sum(
            1 for msg in customer_messages 
            for word in positive_words 
            if word in msg
        )
        base_satisfaction += positive_count * 0.1

        # Negative indicators  
        negative_words = ["angry", "frustrated", "upset", "harassment", "stop calling"]
        negative_count = sum(
            1 for msg in customer_messages 
            for word in negative_words 
            if word in msg
        )
        base_satisfaction -= negative_count * 0.15

        return max(0.0, min(1.0, base_satisfaction))
