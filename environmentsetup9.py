# Create virtual environment
python -m venv nexora_ai
source nexora_ai/bin/activate  # On Windows: nexora_ai\Scripts\activate

# Install core ML libraries
pip install tensorflow==2.13.0
pip install torch torchvision torchaudio
pip install scikit-learn==1.3.0
pip install numpy pandas matplotlib seaborn
pip install opencv-python==4.8.0.76
pip install pillow

# NLP libraries
pip install transformers==4.21.0
pip install spacy==3.6.1
pip install nltk
pip install textblob

# API and web frameworks
pip install flask==2.3.3
pip install fastapi==0.103.1
pip install uvicorn
pip install requests

# Database connectivity
pip install pymongo
pip install psycopg2-binary
pip install redis

# Additional ML tools
pip install joblib
pip install pickle5
pip install streamlit  # For model testing UI