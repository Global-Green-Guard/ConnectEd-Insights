# ConnectEd Insights

**AI-Driven Network Optimization for School Connectivity**

## Overview

ConnectEd Insights is an AI-driven solution aimed at optimizing school connectivity and network infrastructure in underserved regions. Using a combination of artificial intelligence, geospatial data analysis, and dynamic data visualization, this platform enables stakeholders to:

- **Identify connectivity gaps** in schools and communities.
- **Plan optimal new infrastructure** such as cell towers or fiber optic networks.
- **Simulate various network planning scenarios** to assess the impact of different interventions.
- **Monitor network performance** in real-time.

By leveraging open-source technologies such as Python, Streamlit, and LangChain, ConnectEd Insights simplifies complex decision-making processes, empowering non-technical users to undertake data-driven interventions and improve educational connectivity.

## Features

- **Connectivity Gap Analysis:**  
  Identify schools with poor or no internet access using geospatial data.

- **Network Planning Optimization:**  
  Generate recommendations for new infrastructure placement based on cost, coverage, and impact factors.

- **Scenario Simulation:**  
  Simulate the impact of various network interventions and budget adjustments.

- **Real-Time Monitoring:**  
  Visualize network performance data to detect outages and plan maintenance.

- **Generative UI Components:**  
  Utilize interactive maps, charts, and tables to clearly present insights and recommendations.

## Repository Structure

```
ConnectEd-Insights/
├── .gitignore
├── LICENSE
├── README.md
├── app.py
└── modules
    ├── ai_agent.py           # Handles LLM interactions via LangChain for processing natural language queries
    ├── data_processor.py     # Processes geospatial and network data, and generates dummy datasets
    └── visualizations.py     # Contains methods for creating interactive maps and graphs using Streamlit and Altair
```

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Mindscope-AI-Labs/ConnectEd-Insights.git
   cd ConnectEd-Insights
   ```

2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # or
   virtualenv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the Required Packages:**
   If a `requirements.txt` exists:
   ```bash
   pip install -r requirements.txt
   ```
   Otherwise, manually install the necessary packages:
   ```bash
   pip install streamlit pandas altair langchain openai
   ```

4. **Configure API Keys:**
   If you're using the OpenAI integration in `modules/ai_agent.py`, set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   ```
   Alternatively, configure your API key directly in the code as needed.

   - Create .env in the root directory of the project and add your API key as follows:
    ```bash
     OPENAI_API_KEY=your_openai_api_key
    ```


## Running the Application

Start the Streamlit app by running:
```bash
streamlit run app.py
```
This command will launch the application and open it in your default web browser.

## Usage

- **Identify Connectivity Gaps:**  
  Enter queries like:  
  `"Show me schools in rural Kenya with no internet access."`

- **Optimize Network Planning:**  
  Enter queries such as:  
  `"Where should we build new cell towers to maximize coverage for schools in Uganda?"`

- **Simulate Scenarios:**  
  Test interventions with queries like:  
  `"What happens if we increase the budget for satellite internet by 20%?"`

- **Monitor Network Performance:**  
  Analyze network data with queries such as:  
  `"Show me areas with frequent network outages in Nigeria."`

## Contributions

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please reach out via [Mindscope-AI-Labs Support](mailto:paulmwaura254@gmail.com).

---

*Developed as part of the AI for Connectivity Hackathon II: Barcelona Edition.*
