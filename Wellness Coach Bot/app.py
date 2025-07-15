import google.generativeai as genai
import json
import time
import requests
from typing import Dict, Any, Optional 
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

class PersonalWellnessCoach:
    def __init__(self, api_key: str, serper_api_key: str = None):
        """Initialize the Personal Wellness Coach System"""
        genai.configure(api_key=api_key)
        
        self.wellness_model = genai.GenerativeModel("gemini-2.0-flash")
        self.validator_model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Serper API configuration
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.serper_url = "https://google.serper.dev/search"
        
        self.wellness_chat = None
        self.validator_chat = None
        
        self.user_profile = {}
        self.conversation_memory = []
        self.wellness_goals = []
        self.daily_tracking = {}
        self.search_cache = {}  # Cache recent searches
        
        self._setup_conversational_agents()
    
    def _setup_conversational_agents(self):
        """Set up both agents as conversational chatbots"""
        
        wellness_system_prompt = self._get_wellness_system_prompt()
        validator_system_prompt = self._get_validator_system_prompt()
        
        self.wellness_chat = self.wellness_model.start_chat(history=[])
        self.validator_chat = self.validator_model.start_chat(history=[])
        
        self.wellness_chat.send_message(wellness_system_prompt)
        self.validator_chat.send_message(validator_system_prompt)

    def _get_wellness_system_prompt(self) -> str:
        """System prompt for the wellness coach"""
        return """You are Dr. Wellness, a compassionate and knowledgeable AI health and wellness coach with access to current research and information. Your mission is to help people achieve their health goals through personalized guidance, motivation, and evidence-based advice.

Your expertise includes:
- Nutrition and healthy eating habits (with access to current nutritional research)
- Exercise and fitness planning (with latest workout trends and studies)
- Mental health and stress management (current therapeutic approaches)
- Sleep optimization (latest sleep science)
- Habit formation and behavior change (recent behavioral research)
- Preventive health measures (current health guidelines)
- Wellness goal setting and tracking
- Access to real-time health information, research studies, and wellness trends

Your personality:
- Warm, encouraging, and supportive
- Non-judgmental and understanding
- Motivational but realistic
- Evidence-based but accessible (can search for latest research when needed)
- Adaptable to individual needs and preferences
- Always up-to-date with current health information

Guidelines:
- Always prioritize safety and recommend consulting healthcare professionals for medical concerns
- Provide personalized advice based on user's profile and goals
- Use positive reinforcement and celebrate small wins
- Break down complex goals into manageable steps
- Ask clarifying questions to better understand the user's needs
- Be culturally sensitive and inclusive
- Encourage sustainable lifestyle changes over quick fixes
- When you need current information, research studies, or specific health data, you can search for it
- Always cite sources when providing information from searches

When to search:
- User asks about latest research or studies
- Questions about current health trends or guidelines
- Specific nutritional information about foods
- Latest exercise techniques or workout plans
- Current health recommendations from health organizations
- Specific medical conditions or treatments (for general information only)
- Local health resources or facilities
- Product reviews or comparisons for health/fitness items

Remember: You're not a replacement for medical professionals, but a supportive guide for general wellness and healthy lifestyle choices with access to current information."""

    def _get_validator_system_prompt(self) -> str:
        """System prompt for the validator agent"""
        return """You are a wellness conversation validator. Your job is to determine if user input is appropriate for a health and wellness coach conversation.

VALID inputs include:
- Health and wellness questions
- Fitness and exercise queries
- Nutrition and diet questions
- Mental health and stress management
- Sleep and recovery topics
- Goal setting and motivation
- Habit formation and lifestyle changes
- General health concerns (non-medical)

INVALID inputs include:
- Requests for specific medical diagnosis
- Medication advice or dosage questions
- Treatment for serious medical conditions
- Requests for illegal substances or dangerous practices
- Off-topic conversations unrelated to health/wellness
- Inappropriate or harmful content

Response format:
- If valid: "VALID"
- If invalid: "INVALID: [Brief redirect message to wellness topics]"

Be lenient with wellness-related questions and only mark as invalid if clearly inappropriate or potentially harmful."""

    def search_health_info(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for health and wellness information using Serper API"""
        if not self.serper_api_key:
            return {"error": "Serper API key not configured"}
        
        # Check cache first
        cache_key = f"{query}_{num_results}"
        if cache_key in self.search_cache:
            cache_time = self.search_cache[cache_key].get("timestamp", 0)
            # Use cached results if less than 1 hour old
            if time.time() - cache_time < 3600:
                return self.search_cache[cache_key]["results"]
        
        try:
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            # Enhance query for health/wellness context
            enhanced_query = f"{query} health wellness research study"
            
            payload = {
                'q': enhanced_query,
                'num': num_results,
                'gl': 'us',  # Geolocation
                'hl': 'en'   # Language
            }
            
            response = requests.post(self.serper_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            search_results = response.json()
            
            processed_results = self._process_search_results(search_results, query)
            
            self.search_cache[cache_key] = {
                "results": processed_results,
                "timestamp": time.time()
            }
            
            return processed_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return {"error": f"Search failed: {str(e)}"}
    
    def _process_search_results(self, raw_results: Dict, original_query: str) -> Dict[str, Any]:
        """Process and filter search results for health relevance"""
        processed = {
            "query": original_query,
            "results": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        organic_results = raw_results.get("organic", [])
        
        for result in organic_results:
            trusted_domains = [
                'nih.gov', 'cdc.gov', 'who.int', 'mayo clinic', 'harvard health',
                'webmd.com', 'healthline.com', 'medicalnewstoday.com', 
                'pubmed.ncbi.nlm.nih.gov', 'nhs.uk', 'cleveland clinic'
            ]
            
            source_url = result.get("link", "").lower()
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            
            # Prioritize trusted sources
            trust_score = 1
            for domain in trusted_domains:
                if domain in source_url or domain in title.lower():
                    trust_score = 5
                    break
            
            processed_result = {
                "title": title,
                "url": result.get("link"),
                "snippet": snippet,
                "trust_score": trust_score,
                "source": self._extract_domain(result.get("link", ""))
            }
            
            processed["results"].append(processed_result)
        
        # Sort by trust score
        processed["results"].sort(key=lambda x: x["trust_score"], reverse=True)
        
        return processed
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '')
        except:
            return "Unknown"
    
    def _should_search(self, user_input: str) -> bool:
        """Determine if the user input requires a search"""
        search_triggers = [
            'latest', 'recent', 'current', 'new study', 'research shows',
            'what does science say', 'studies on', 'research on',
            'latest guidelines', 'current recommendations', 'recent findings',
            'what are the benefits of', 'nutritional information',
            'calories in', 'is it healthy', 'side effects',
            'how to', 'best way to', 'most effective',
            'compare', 'vs', 'difference between',
            'local', 'near me', 'in my area'
        ]
        
        user_lower = user_input.lower()
        return any(trigger in user_lower for trigger in search_triggers)

    def _is_valid_input(self, user_input: str) -> tuple[bool, str]:
        """Quick validation check using the validator agent"""
        try:
            validation_prompt = f"""Validate this user input: "{user_input}"
            
Previous conversation context: {self._get_recent_context()}

Is this appropriate for a wellness coach?"""

            response = self.validator_chat.send_message(validation_prompt)
            result = response.text.strip()
            
            if result.startswith("VALID"):
                return True, ""
            elif result.startswith("INVALID"):
                redirect_msg = result.replace("INVALID:", "").strip()
                if not redirect_msg:
                    redirect_msg = "I'm here to help with your health and wellness journey! What would you like to know about nutrition, fitness, mental health, or healthy habits?"
                return False, redirect_msg
            else:
                return True, ""
                
        except Exception as e:
            print(f"Validation error: {e}")
            return True, ""

    def _get_recent_context(self) -> str:
        """Get recent conversation context for validation"""
        if len(self.conversation_memory) <= 3:
            return json.dumps(self.conversation_memory)
        else:
            return json.dumps(self.conversation_memory[-3:])  # Last 3 exchanges

    def _add_to_memory(self, user_msg: str, agent_response: str):
        """Add exchange to conversation memory"""
        self.conversation_memory.append({
            "user": user_msg,
            "agent": agent_response,
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        if len(self.conversation_memory) > 20:
            self.conversation_memory = self.conversation_memory[-20:]

    def update_user_profile(self, profile_data: Dict[str, Any]):
        """Update user profile information"""
        self.user_profile.update(profile_data)
        
    def add_wellness_goal(self, goal: str, target_date: str = None, category: str = "general"):
        """Add a wellness goal for the user"""
        goal_data = {
            "goal": goal,
            "category": category,
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "target_date": target_date,
            "status": "active",
            "progress": 0
        }
        self.wellness_goals.append(goal_data)
        
    def track_daily_metric(self, metric: str, value: Any, date: str = None):
        """Track daily wellness metrics"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        if date not in self.daily_tracking:
            self.daily_tracking[date] = {}
            
        self.daily_tracking[date][metric] = value

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of user's wellness progress"""
        return {
            "active_goals": len([g for g in self.wellness_goals if g["status"] == "active"]),
            "completed_goals": len([g for g in self.wellness_goals if g["status"] == "completed"]),
            "tracking_days": len(self.daily_tracking),
            "recent_activity": list(self.daily_tracking.keys())[-7:] if self.daily_tracking else []
        }

    def chat(self, user_input: str) -> str:
        """Main chat method - handles user input and returns response"""
        
        user_input = user_input.strip()
        if not user_input:
            return "I'm here to support your wellness journey! What would you like to talk about today?"
        
        is_valid, validation_msg = self._is_valid_input(user_input)
        
        if not is_valid:
            return validation_msg
        
        try:
            search_results = None
            if self._should_search(user_input):
                print("🔍 Searching for latest health information...")
                search_results = self.search_health_info(user_input)
                
                if "error" not in search_results:
                    print(f"✅ Found {len(search_results.get('results', []))} relevant sources")
            
            context_prompt = f"""User message: "{user_input}"

Recent conversation: {self._get_recent_context()}

User Profile: {json.dumps(self.user_profile) if self.user_profile else "Not yet established"}

Current Wellness Goals: {json.dumps(self.wellness_goals) if self.wellness_goals else "None set"}

Recent Progress: {json.dumps(self.get_progress_summary())}

Daily Tracking Data (last 7 days): {json.dumps(dict(list(self.daily_tracking.items())[-7:]))}"""

            if search_results and "error" not in search_results:
                context_prompt += f"""

CURRENT RESEARCH & INFORMATION (from search):
Query: {search_results.get('query', 'N/A')}
Sources found: {len(search_results.get('results', []))}

Search Results:
{json.dumps(search_results.get('results', [])[:3], indent=2)}

Please incorporate this current information into your response when relevant. Always cite sources when using search information."""

            context_prompt += """

Please respond as Dr. Wellness, keeping in mind our previous conversations and the user's wellness journey. Be supportive, personalized, and actionable in your response. If you used search results, mention the sources and cite them appropriately."""

            response = self.wellness_chat.send_message(context_prompt)
            agent_response = response.text
            
            # Add search info to response if sources were used
            if search_results and "error" not in search_results and search_results.get('results'):
                agent_response += f"\n\n📚 Sources: Based on current research from {len(search_results['results'])} sources including {', '.join(set([r['source'] for r in search_results['results'][:3]]))}."
            
            # Add to conversation memory
            self._add_to_memory(user_input, agent_response)
            
            return agent_response
            
        except Exception as e:
            error_msg = "I'm having a small technical hiccup. Could you try asking that again? I'm here to help with your wellness journey!"
            print(f"Chat error: {e}")
            return error_msg

    def manual_search(self, query: str) -> str:
        """Manual search function for users to trigger searches"""
        print(f"🔍 Searching for: {query}")
        results = self.search_health_info(query)
        
        if "error" in results:
            return f"❌ Search error: {results['error']}"
        
        if not results.get('results'):
            return "No relevant health information found for your query."
        
        response = f"🔍 Search Results for '{query}':\n\n"
        
        for i, result in enumerate(results['results'][:5], 1):
            trust_indicator = "⭐" * result.get('trust_score', 1)
            response += f"{i}. {result['title']} {trust_indicator}\n"
            response += f"   Source: {result['source']}\n"
            response += f"   {result['snippet'][:150]}...\n"
            response += f"   URL: {result['url']}\n\n"
        
        return response

    def get_conversation_history(self) -> list:
        """Get the full conversation history"""
        return self.conversation_memory.copy()

    def clear_conversation(self):
        """Clear conversation history and start fresh"""
        self.conversation_memory = []
        self._setup_conversational_agents()

    def save_session(self, filename: str = None):
        """Save complete session to file"""
        if filename is None:
            filename = f"wellness_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        try:
            save_data = {
                "user_profile": self.user_profile,
                "conversation_memory": self.conversation_memory,
                "wellness_goals": self.wellness_goals,
                "daily_tracking": self.daily_tracking,
                "session_timestamp": time.time(),
                "session_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            return f"Wellness session saved to {filename}"
        except Exception as e:
            return f"Error saving session: {e}"

    def load_session(self, filename: str):
        """Load session from file"""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)
            
            self.user_profile = save_data.get("user_profile", {})
            self.conversation_memory = save_data.get("conversation_memory", [])
            self.wellness_goals = save_data.get("wellness_goals", [])
            self.daily_tracking = save_data.get("daily_tracking", {})
            
            return f"Wellness session loaded from {filename}"
        except Exception as e:
            return f"Error loading session: {e}"

    def quick_setup_profile(self):
        """Interactive profile setup"""
        print("\n🌟 Let's set up your wellness profile!")
        print("This helps me provide personalized guidance. You can skip any question.")
        
        profile = {}
        
        # Basic info
        age = input("What's your age? (optional): ").strip()
        if age: 
            profile["age"] = age
        
        activity_level = input("How would you describe your current activity level? (sedentary/lightly active/moderately active/very active): ").strip()
        if activity_level: 
            profile["activity_level"] = activity_level
        
        # Health goals
        primary_goal = input("What's your main wellness goal? (e.g., lose weight, build muscle, reduce stress, improve sleep): ").strip()
        if primary_goal: 
            profile["primary_goal"] = primary_goal
            self.add_wellness_goal(primary_goal, category="primary")
        
        # Dietary preferences
        dietary_pref = input("Any dietary preferences or restrictions? (e.g., vegetarian, keto, allergies): ").strip()
        if dietary_pref: 
            profile["dietary_preferences"] = dietary_pref
        
        # Health conditions
        health_notes = input("Any health conditions I should be aware of? (optional): ").strip()
        if health_notes: 
            profile["health_notes"] = health_notes
        
        self.update_user_profile(profile)
        print(" Profile created! I'll use this information to provide personalized guidance.")
        
        return profile


def main():
    """Main interactive chat loop"""
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    if not API_KEY:
        print(" Please set your GEMINI_API_KEY environment variable")
        return
        
    if not SERPER_API_KEY:
        print(" SERPER_API_KEY not found. Search functionality will be limited.")
        print("Get your API key from: https://serper.dev/")
    
    try:
        print(" Initializing Dr. Wellness with Internet Search...")
        coach = PersonalWellnessCoach(API_KEY, SERPER_API_KEY)
        
        print("\n" + "="*70)
        print(" DR. WELLNESS - Your Personal Health & Wellness Coach")
        print(" NOW WITH REAL-TIME HEALTH RESEARCH & INFORMATION!")
        print("="*70)
        print("Welcome! I'm Dr. Wellness, your AI-powered health and wellness coach")
        print("with access to the latest health research and current information!")
        print("\n I can help with:")
        print("   • Nutrition and healthy eating (with current research)")
        print("   • Exercise and fitness planning (latest workout trends)")
        print("   • Mental health and stress management • Sleep optimization")
        print("   • Habit formation • Goal setting and tracking")
        print("   • Latest health studies and recommendations")
        print("\n Commands:")
        print("   • 'setup' - Create your wellness profile")
        print("   • 'goals' - View/manage your wellness goals")
        print("   • 'track' - Log daily wellness metrics")
        print("   • 'progress' - View your wellness progress")
        print("   • 'search [query]' - Search for health information")
        print("   • 'save' - Save your session • 'load [filename]' - Load previous session")
        print("   • 'clear' - Start fresh • 'history' - View recent conversations")
        print("   • 'exit' or 'quit' - End session")
        print("\n Search triggers automatically when you ask about:")
        print("   • Latest research, current studies, recent findings")
        print("   • Nutritional information, calories, health benefits")
        print("   • Best practices, guidelines, recommendations")
        print("-" * 70)
        
        # Initial greeting
        greeting = coach.chat("Hello! I'm excited to be your wellness coach with access to current health research. How can I help you on your health journey today?")
        print(f"\n🩺 Dr. Wellness: {greeting}")
        
        while True:
            try:
                user_input = input("\n You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(" Dr. Wellness: It's been wonderful supporting your wellness journey! Remember, every small step counts. Keep up the great work and feel free to return anytime. Stay healthy! 🌟")
                    break
                
                elif user_input.lower() == 'setup':
                    coach.quick_setup_profile()
                    response = coach.chat("I've updated your profile! Now I can provide more personalized guidance. What would you like to focus on first?")
                    print(f"\n Dr. Wellness: {response}")
                    continue
                
                elif user_input.lower() == 'goals':
                    if coach.wellness_goals:
                        print("\n Your Wellness Goals:")
                        for i, goal in enumerate(coach.wellness_goals, 1):
                            status_emoji = "✅" if goal["status"] == "completed" else "🎯"
                            print(f"   {status_emoji} {goal['goal']} ({goal['category']}) - {goal['status']}")
                    else:
                        print("\n No goals set yet. Let's create some!")
                        goal = input("What wellness goal would you like to set? ").strip()
                        if goal:
                            coach.add_wellness_goal(goal)
                            print(f" Goal added: {goal}")
                    continue
                
                elif user_input.lower() == 'track':
                    print("\n Daily Wellness Tracking")
                    print("What would you like to track today?")
                    print("Examples: water intake, steps, mood (1-10), sleep hours, exercise minutes")
                    
                    metric = input("Metric: ").strip()
                    if metric:
                        value = input(f"Value for {metric}: ").strip()
                        if value:
                            coach.track_daily_metric(metric, value)
                            print(f" Tracked: {metric} = {value}")
                    continue
                
                elif user_input.lower() == 'progress':
                    summary = coach.get_progress_summary()
                    print("\nYour Wellness Progress:")
                    print(f"   Active Goals: {summary['active_goals']}")
                    print(f"   Completed Goals: {summary['completed_goals']}")
                    print(f"   Days Tracked: {summary['tracking_days']}")
                    if summary['recent_activity']:
                        print(f"   Recent Activity: {', '.join(summary['recent_activity'])}")
                    continue
                
                elif user_input.lower().startswith('search'):
                    search_query = user_input.replace('search', '').strip()
                    if search_query:
                        result = coach.manual_search(search_query)
                        print(f"\n{result}")
                    else:
                        search_query = input("What would you like to search for? ").strip()
                        if search_query:
                            result = coach.manual_search(search_query)
                            print(f"\n{result}")
                    continue
                
                elif user_input.lower() == 'save':
                    result = coach.save_session()
                    print(f"\n💾 {result}")
                    continue
                
                elif user_input.lower().startswith('load'):
                    filename = user_input.replace('load', '').strip()
                    if not filename:
                        filename = input("Enter filename: ").strip()
                    if filename:
                        result = coach.load_session(filename)
                        print(f"\n📁 {result}")
                    continue
                
                elif user_input.lower() == 'clear':
                    coach.clear_conversation()
                    print("\n🔄 Conversation cleared! Starting fresh.")
                    greeting = coach.chat("Let's start fresh! How can I help you with your wellness journey today?")
                    print(f"\n🩺 Dr. Wellness: {greeting}")
                    continue
                
                elif user_input.lower() == 'history':
                    history = coach.get_conversation_history()
                    print(f"\n📜 Recent Conversations ({len(history)} total):")
                    for i, exchange in enumerate(history[-5:], 1):  # Show last 5
                        print(f"   {i}. You: {exchange['user'][:60]}...")
                        print(f"      Dr. Wellness: {exchange['agent'][:60]}...")
                    continue
                
                if not user_input:
                    continue
                
                response = coach.chat(user_input)
                print(f"\n🩺 Dr. Wellness: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🩺 Dr. Wellness: Take care of yourself! Remember, wellness is a journey, not a destination. Come back anytime! 🌟")
                break
            except Exception as e:
                print(f"\n⚠️ Oops, something went wrong: {e}")
                print("Let's try that again!")
                
    except Exception as e:
        print(f"❌ Error starting Dr. Wellness: {e}")
        print("Please check your API keys and try again.")

if __name__ == "__main__":
    main()
