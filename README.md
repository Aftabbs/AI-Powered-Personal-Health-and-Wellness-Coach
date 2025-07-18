# Dr. Wellness - Personal Health & Wellness Coach
 
<img width="307" height="94" alt="image" src="https://github.com/user-attachments/assets/b7cfedb9-9e39-42e5-829a-61ab2e74aaa5" />



A conversational AI health and wellness coach powered by Google's Gemini AI with real-time internet search capabilities using Serper API. Dr. Wellness provides personalized guidance, tracks progress, and accesses current health research to support your wellness journey.

## Features
 
### Core Functionality
- **Personalized Wellness Coaching**: Tailored advice based on user profiles and goals
- **Real-time Health Research**: Access to current studies and health information via Serper API
- **Goal Setting & Tracking**: Set, monitor, and achieve wellness objectives
- **Daily Metrics Tracking**: Log and analyze health metrics like water intake, exercise, mood, sleep
- **Conversation Memory**: Maintains context across sessions for personalized interactions
- **Progress Analytics**: View achievements, patterns, and wellness trends
- **Session Persistence**: Save and load complete wellness journeys

### Smart Search Integration
- **Automatic Search Triggers**: Searches for latest research when users ask about current studies
- **Manual Search Commands**: Direct search functionality for health information
- **Source Verification**: Prioritizes trusted health sources (NIH, CDC, Mayo Clinic, etc.)
- **Result Caching**: Intelligent caching to optimize API usage
- **Citation Integration**: Automatic source citations in responses

### Validation & Safety
- **Input Validation**: Dual-agent system ensures conversations stay health-focused
- **Medical Disclaimers**: Always recommends consulting healthcare professionals for medical concerns
- **Evidence-based Advice**: Provides scientifically-backed wellness recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Serper API key (optional, for search functionality)

### Dependencies
Install required packages using pip:

```bash
pip install google-generativeai requests python-dotenv
```

### API Keys Setup

1. **Google Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create or use existing API key

2. **Serper API Key** (optional but recommended):
   - Visit [Serper.dev](https://serper.dev/)
   - Sign up and get your API key

3. **Environment Variables**:
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

## Usage

### Basic Usage
Run the application:
```bash
python wellness_coach.py
```

### Interactive Commands

#### Profile Management
- `setup` - Create your personalized wellness profile
- `goals` - View and manage wellness goals
- `track` - Log daily wellness metrics
- `progress` - View progress summary and achievements

#### Search Functionality
- `search [query]` - Manual search for health information
- Auto-search triggers on keywords like "latest research", "current studies", "benefits of"

#### Session Management
- `save` - Save current session to file
- `load [filename]` - Load previous session
- `clear` - Start fresh conversation
- `history` - View recent conversation exchanges

#### General
- `exit` or `quit` - End session

### Example Interactions

#### Automatic Search Triggers
```
You: "What are the latest studies on intermittent fasting?"
Dr. Wellness: [Searches automatically and provides current research with citations]
```

#### Manual Health Information Search
```
You: "search benefits of meditation"
Dr. Wellness: [Returns formatted search results with trust scores]
```

#### Goal Setting and Tracking
```
You: "I want to lose 10 pounds in 3 months"
Dr. Wellness: [Creates goal, provides personalized plan, sets up tracking]
```

## Configuration

### Search Behavior Customization
Modify search triggers in the `_should_search()` method:
```python
search_triggers = [
    'latest', 'recent', 'current', 'new study',
    'research shows', 'studies on', 'benefits of',
    # Add your custom triggers
]
```

### Trusted Health Sources
Configure trusted domains in `_process_search_results()`:
```python
trusted_domains = [
    'nih.gov', 'cdc.gov', 'who.int', 'mayo clinic',
    # Add additional trusted sources
]
```

## File Structure

```
wellness-coach/
├── app.py          # Main application file
├── .env                       # Environment variables (create this)
├── README.md                  # This file
└── sessions/                  # Saved session files (auto-created)
    ├── wellness_session_20240115_143022.json
    └── ...
```

## Data Storage

### Session Files
Sessions are saved as JSON files containing:
- User profile information
- Conversation history
- Wellness goals and progress
- Daily tracking data
- Timestamps and metadata

### Data Privacy
- All data is stored locally
- No personal information is sent to external services except search queries
- Users control their data through save/load functionality

## Limitations and Disclaimers

### Medical Disclaimer
Dr. Wellness is not a replacement for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

### API Limitations
- **Gemini API**: Subject to Google's usage limits and policies
- **Serper API**: Free tier includes 2,500 searches per month
- Search results depend on internet connectivity and API availability

### Scope of Advice
- Provides general wellness guidance only
- Does not diagnose medical conditions
- Does not recommend specific medications or treatments
- Focuses on lifestyle, nutrition, exercise, and general health

## Contributing

### Reporting Issues
1. Check existing issues on GitHub
2. Provide detailed description of the problem
3. Include error messages and steps to reproduce

### Feature Requests
1. Open an issue with the "enhancement" label
2. Describe the proposed feature and use case
3. Explain how it would benefit users

### Development
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **Google Gemini AI**: Powers the conversational AI capabilities
- **Serper API**: Provides real-time search functionality
- **Health Information Sources**: NIH, CDC, WHO, Mayo Clinic, and other trusted health organizations

## Support

For questions, issues, or suggestions:
1. Check the documentation
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. For general wellness questions, use the bot itself

## Version History

### v1.0.0 (Current)
- Initial release with full wellness coaching capabilities
- Integrated Serper API for real-time health research
- Comprehensive goal setting and tracking system
- Session persistence and conversation memory
- Dual-agent validation system

---

**Note**: This is an AI-powered wellness tool designed to provide general health and wellness guidance. Always consult healthcare professionals for medical advice and treatment decisions.
