
#!/usr/bin/env python3
"""
AI-Automated Testing Platform for Voice Agents
Self-Correcting Voice Agent System for Debt Collection

This system automatically generates loan defaulter personalities, tests voice agents,
and implements self-correction mechanisms to improve agent performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import numpy as np

# Import our custom modules
from modules.persona_generator import PersonaGenerator, DebtorPersona
from modules.conversation_simulator import ConversationSimulator, TestScenario
from modules.automated_tester import VoiceAgentTester
from modules.self_correcting_agent import SelfCorrectingVoiceAgent

def main():
    st.set_page_config(
        page_title="Voice Agent Testing Platform",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ AI-Automated Voice Agent Testing Platform")
    st.markdown("### Self-Correcting Debt Collection Voice Agents")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Generate Personas", "Test Voice Agent", "Self-Correcting Agent", "View Results"]
    )

    if page == "Overview":
        show_overview()
    elif page == "Generate Personas":
        show_persona_generator()
    elif page == "Test Voice Agent":
        show_agent_tester()
    elif page == "Self-Correcting Agent":
        show_self_correcting_agent()
    elif page == "View Results":
        show_results()

def show_overview():
    st.header("Platform Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Key Features")
        st.markdown("""
        - **Automated Persona Generation**: Creates diverse loan defaulter personalities
        - **Conversation Simulation**: Simulates realistic debt collection conversations
        - **Performance Metrics**: Tracks repetition, negotiation, relevance, compliance, empathy
        - **Self-Correction**: Automatically improves agent prompts based on test results
        - **Comprehensive Testing**: Handles multiple scenarios and edge cases
        """)

    with col2:
        st.subheader("📊 Metrics Tracked")
        st.markdown("""
        - **Repetition Score**: Measures if bot repeats itself (lower is better)
        - **Negotiation Effectiveness**: Ability to negotiate payment arrangements
        - **Response Relevance**: How well responses match customer needs
        - **Compliance Score**: Adherence to debt collection regulations
        - **Empathy Score**: Emotional intelligence in responses
        - **Resolution Rate**: Percentage of successful outcomes
        """)

    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Generate Personas**: Create diverse test personalities in the "Generate Personas" section
    2. **Test Your Agent**: Input your voice agent prompt and run automated tests
    3. **Review Results**: Analyze performance metrics and identify improvement areas
    4. **Self-Correct**: Use the automated improvement system to enhance your agent
    """)

def show_persona_generator():
    st.header("👥 Persona Generator")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        num_personas = st.number_input("Number of personas to generate:", min_value=1, max_value=100, value=10)
    with col2:
        selected_scenarios = st.multiselect(
            "Target scenarios (optional):",
            [scenario.value for scenario in TestScenario],
            default=[]
        )

    if st.button("Generate Personas"):
        with st.spinner("Generating personas..."):
            generator = PersonaGenerator()
            scenarios = [TestScenario(s) for s in selected_scenarios] if selected_scenarios else None
            personas = generator.generate_batch(num_personas, scenarios)

            # Store in session state
            st.session_state['personas'] = personas

            # Display results
            st.success(f"Generated {len(personas)} personas!")

            # Show summary
            df_data = []
            for persona in personas:
                df_data.append({
                    'Name': persona.name,
                    'Age': persona.age,
                    'Debt Amount': f"${persona.debt_amount:,.2f}",
                    'Days Past Due': persona.days_past_due,
                    'Employment': persona.employment_status,
                    'Communication Style': persona.communication_style,
                    'Traits': ', '.join(persona.personality_traits[:3]),  # Show first 3 traits
                    'Negotiation Likelihood': f"{persona.negotiation_likelihood:.2f}",
                    'Stress Level': f"{persona.financial_stress_level}/10"
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df)

            # Show analytics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_debt = np.mean([p.debt_amount for p in personas])
                st.metric("Average Debt", f"${avg_debt:,.2f}")

            with col2:
                avg_negotiation = np.mean([p.negotiation_likelihood for p in personas])
                st.metric("Avg Negotiation Likelihood", f"{avg_negotiation:.2f}")

            with col3:
                avg_stress = np.mean([p.financial_stress_level for p in personas])
                st.metric("Avg Stress Level", f"{avg_stress:.1f}/10")

def show_agent_tester():
    st.header("🧪 Voice Agent Tester")

    # Agent prompt input
    st.subheader("Agent Prompt")
    default_prompt = """You are Sarah, a professional debt collection specialist at Financial Recovery Services.

Your role is to:
- Contact customers about overdue payments respectfully and professionally
- Follow all FDCPA compliance requirements
- Show empathy while being firm about payment obligations
- Offer flexible payment solutions when appropriate
- Maintain detailed records of all interactions

Always include required compliance statements and avoid threatening language."""

    agent_prompt = st.text_area(
        "Enter your voice agent prompt:",
        value=default_prompt,
        height=200
    )

    # Test configuration
    col1, col2 = st.columns(2)
    with col1:
        test_count = st.number_input("Number of test conversations:", min_value=5, max_value=100, value=20)
    with col2:
        test_scenarios = st.multiselect(
            "Test scenarios:",
            [scenario.value for scenario in TestScenario],
            default=[TestScenario.PAYMENT_REMINDER.value, TestScenario.FIRST_CONTACT.value]
        )

    if st.button("Run Tests"):
        if not agent_prompt.strip():
            st.error("Please enter an agent prompt")
            return

        if not test_scenarios:
            st.error("Please select at least one test scenario")
            return

        with st.spinner(f"Running {test_count} test conversations..."):
            tester = VoiceAgentTester()
            scenarios = [TestScenario(s) for s in test_scenarios]

            results = tester.run_test_suite(
                agent_prompt=agent_prompt,
                test_count=test_count,
                scenarios=scenarios
            )

            # Store results
            st.session_state['test_results'] = results

            # Display results
            display_test_results(results)

def display_test_results(results):
    st.success(f"Completed {results['total_tests']} tests!")

    metrics = results['aggregate_metrics']

    # Overall score
    overall_score = metrics['overall_score']
    st.metric("Overall Performance Score", f"{overall_score:.3f}", 
              help="Weighted average of all metrics (higher is better)")

    # Metrics dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Repetition Score", f"{metrics['repetition_score']:.3f}", 
                  help="Lower is better - measures if agent repeats itself")
        st.metric("Negotiation Effectiveness", f"{metrics['negotiation_effectiveness']:.3f}",
                  help="Higher is better - ability to negotiate solutions")

    with col2:
        st.metric("Response Relevance", f"{metrics['response_relevance']:.3f}",
                  help="Higher is better - how well responses match customer needs")
        st.metric("Compliance Score", f"{metrics['compliance_score']:.3f}",
                  help="Higher is better - adherence to regulations")

    with col3:
        st.metric("Empathy Score", f"{metrics['empathy_score']:.3f}",
                  help="Higher is better - emotional intelligence shown")
        st.metric("Resolution Rate", f"{metrics['resolution_rate']:.3f}",
                  help="Percentage of conversations that reached resolution")

    # Performance by scenario
    st.subheader("📊 Performance by Scenario")
    scenario_df = pd.DataFrame([
        {'Scenario': scenario, 'Average Score': data['average_score'], 'Test Count': data['test_count']}
        for scenario, data in results['scenario_performance'].items()
    ])

    fig = px.bar(scenario_df, x='Scenario', y='Average Score', 
                 title="Performance by Test Scenario")
    st.plotly_chart(fig)

    # Recommendations
    st.subheader("💡 Recommendations")
    for i, rec in enumerate(results['recommendations'], 1):
        if rec.startswith("CRITICAL"):
            st.error(f"{i}. {rec}")
        elif rec.startswith("HIGH PRIORITY"):
            st.warning(f"{i}. {rec}")
        else:
            st.info(f"{i}. {rec}")

def show_self_correcting_agent():
    st.header("🔄 Self-Correcting Voice Agent")

    st.markdown("""
    This system automatically improves your voice agent through iterative testing and prompt optimization.
    """)

    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        target_score = st.slider("Target Score", 0.5, 1.0, 0.8, 0.05)
    with col2:
        max_iterations = st.number_input("Max Iterations", 1, 10, 5)
    with col3:
        tests_per_iteration = st.number_input("Tests per Iteration", 10, 50, 20)

    # Initial prompt
    initial_prompt = st.text_area(
        "Initial Agent Prompt:",
        value="""You are a debt collection agent. Contact customers about overdue payments. 
Be professional and try to collect payment or set up payment arrangements.""",
        height=150
    )

    if st.button("Start Self-Improvement Process"):
        if not initial_prompt.strip():
            st.error("Please enter an initial prompt")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        # Create self-correcting agent
        agent = SelfCorrectingVoiceAgent(
            initial_prompt=initial_prompt,
            target_score=target_score,
            max_iterations=max_iterations
        )

        try:
            # Run improvement cycle with progress updates
            iteration = 0
            agent.iteration_count = 0

            while agent.iteration_count < agent.max_iterations:
                iteration += 1
                status_text.text(f"Running iteration {iteration}/{max_iterations}...")
                progress_bar.progress(iteration / max_iterations)

                # Test current prompt
                results = agent.tester.run_test_suite(
                    agent_prompt=agent.current_prompt,
                    test_count=tests_per_iteration
                )

                current_score = results['aggregate_metrics']['overall_score']

                # Store results
                iteration_data = {
                    'iteration': iteration,
                    'prompt': agent.current_prompt,
                    'score': current_score,
                    'results': results,
                    'timestamp': datetime.now()
                }
                agent.improvement_history.append(iteration_data)

                # Update metrics display
                with metrics_container.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Iteration", iteration)
                    with col2:
                        st.metric("Current Score", f"{current_score:.3f}")
                    with col3:
                        if len(agent.improvement_history) > 1:
                            prev_score = agent.improvement_history[-2]['score']
                            improvement = current_score - prev_score
                            st.metric("Score Change", f"{improvement:+.3f}")
                        else:
                            st.metric("Score Change", "N/A")

                # Check if target reached
                if current_score >= target_score:
                    status_text.text(f"✅ Target score reached! Final score: {current_score:.3f}")
                    break

                # Generate improved prompt
                new_prompt = agent._improve_prompt(agent.current_prompt, results)

                if new_prompt == agent.current_prompt:
                    status_text.text("No further improvements possible.")
                    break

                agent.current_prompt = new_prompt
                agent.iteration_count += 1

            progress_bar.progress(1.0)

            # Generate and display final report
            final_report = agent._generate_final_report()
            st.session_state['improvement_results'] = final_report

            # Display results
            display_improvement_results(final_report)

        except Exception as e:
            st.error(f"Error during improvement process: {str(e)}")

def display_improvement_results(results):
    st.success("Self-improvement process completed!")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Score", f"{results['initial_score']:.3f}")
    with col2:
        st.metric("Final Score", f"{results['final_score']:.3f}")
    with col3:
        st.metric("Improvement", f"{results['improvement']:+.3f}")
    with col4:
        st.metric("Iterations", results['iterations_used'])

    # Score progression chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(results['score_progression']) + 1)),
        y=results['score_progression'],
        mode='lines+markers',
        name='Score Progression',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Score Progression Over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Overall Score",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig)

    # Final optimized prompt
    st.subheader("🎯 Final Optimized Prompt")
    st.text_area("Optimized Agent Prompt:", value=results['final_prompt'], height=300, disabled=True)

    # Target achievement status
    if results['target_achieved']:
        st.success(f"🎉 Target score of {results.get('target_score', 'N/A')} achieved!")
    else:
        st.warning(f"Target score not reached. Consider running more iterations or adjusting the target.")

def show_results():
    st.header("📈 Results Dashboard")

    # Show stored test results
    if 'test_results' in st.session_state:
        st.subheader("Latest Test Results")
        display_test_results(st.session_state['test_results'])

    # Show improvement results
    if 'improvement_results' in st.session_state:
        st.subheader("Self-Improvement Results")
        display_improvement_results(st.session_state['improvement_results'])

    if 'test_results' not in st.session_state and 'improvement_results' not in st.session_state:
        st.info("No results available yet. Run some tests to see results here!")

if __name__ == "__main__":
    main()
