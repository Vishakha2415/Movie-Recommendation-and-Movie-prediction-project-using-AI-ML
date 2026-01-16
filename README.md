# Movie-Recommendation-and-Movie-prediction-project-using-AI-ML
AI/ML movie system using IMDb data. Recommends films via content-based filtering (genre, cast, keywords) and predicts ratings/success with supervised learning. Built in Python with Pandas, Scikit-learn,
# 🎬 Movie Recommendation & Prediction System (AI/ML)

A comprehensive AI-powered platform that combines movie success prediction with personalized recommendation capabilities using machine learning algorithms.

## 📁 Project Structure


final/
│
├── frontend/
│   └── index.html          # Complete web interface (HTML, CSS, JS)
│
|
|
└── backend/
    ├── app.py              # Flask server with REST API endpoints
    ├── movie.py            # Core ML models and recommendation engine
    └── movies_dataset.csv  # IMDb movie dataset


## ✨ Key Features

### 🎯 Prediction Module
- **Movie Success Prediction**: Predict whether a movie will be a HIT or FLOP
- **Dual Prediction Methods**:
  1. By Movie Name: Search existing movies in dataset
  2. By Details: Analyze director, actors, genre combinations
- **AI Analysis**: Uses Random Forest classifier for predictions
- **Detailed Reports**: Confidence scores, rating predictions, key factors

### 🎬 Recommendation Engine
- **Personalized Suggestions**: Content-based filtering using TF-IDF
- **Interactive Rating System**: Like/Dislike ratings to build user profile
- **Smart Search**: Filter by genre, director, actors, or keywords
- **Progress Tracking**: Visual feedback for rating progress

### 🎨 User Interface
- **Modern Design**: Gradient aesthetics with responsive layout
- **Real-time Updates**: Instant feedback on interactions
- **Mobile Responsive**: Works across all device sizes
- **Intuitive Navigation**: Separate tabs for prediction and recommendation

## 🛠️ Technology Stack

### Backend (Python)
- **Flask**: Lightweight web framework
- **Scikit-learn**: ML algorithms (Random Forest, TF-IDF)
- **Pandas & NumPy**: Data processing and analysis
- **Flask-CORS**: Cross-origin resource sharing

### Frontend
- **HTML5/CSS3**: Modern semantic markup with CSS Grid/Flexbox
- **Vanilla JavaScript**: No external frameworks required
- **Font Awesome**: Icon toolkit
- **Responsive Design**: Mobile-first approach

### Machine Learning Models
- **Random Forest Classifier**: For success prediction (Hit/Flop)
- **Content-Based Filtering**: For movie recommendations
- **Feature Engineering**: Genre encoding, director analysis, actor popularity
- **Similarity Scoring**: Cosine similarity for recommendations

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Web browser (Chrome/Firefox recommended)

### Step-by-Step Setup

1. **Navigate to project directory:**
bash
cd final/backend


2. **Create virtual environment (optional but recommended):**
bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


3. **Install required packages:**
bash
pip install flask flask-cors pandas numpy scikit-learn


4. **Prepare dataset:**
- Place your movies_dataset.csv in the backend` folder
- Ensure CSV has required columns (see Dataset Format section)

5. **Run the application:**
bash
python app.py


6. **Access the system:**
- Open browser and go to: http://localhost:5000

## 📊 Dataset Format

The system expects a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Name` | Movie title | "The Dark Knight" |
| `Year` | Release year | 2008 |
| `Duration` | Runtime in minutes | 152 |
| `Genre` | Primary genre | "Action" |
| `Rating` | IMDb rating (1-10) | 9.0 |
| `Votes` | Number of votes | 2500000 |
| `Director` | Director name | "Christopher Nolan" |
| `Actor 1` | Lead actor 1 | "Christian Bale" |
| `Actor 2` | Lead actor 2 | "Heath Ledger" |
| `Actor 3` | Lead actor 3 | "Aaron Eckhart" |

**Note**: If dataset not found, system creates sample data for testing.

## 🔧 How to Use

### Movie Predictor (Tab 1)

1. **Predict by Movie Name:**
   - Enter movie title (e.g., "Inception")
   - Optional: Add release year for better accuracy
   - Click "Predict Success"

2. **Predict by Details:**
   - Fill in director, lead actor, and genre
   - Optional: Add second actor
   - Click "Analyze & Predict"

3. **View Results:**
   - Success prediction (HIT/FLOP)
   - Predicted rating (1-10)
   - Confidence percentage
   - Key influencing factors

### Movie Recommender (Tab 2)

1. **Rate Movies:**
   - Browse displayed movies
   - Click "I Like This" or "Not For Me"
   - Rate at least 3 movies

2. **Search Movies:**
   - Use search bar for specific movies
   - Filter by genre using tags
   - Rate search results

3. **Get Recommendations:**
   - After 3+ ratings, click "Get Recommendations"
   - View personalized movie suggestions
   - See match scores and reasons

## 🧠 Machine Learning Details

### Prediction Model
1. **Data Preparation:**
   - Clean and preprocess movie data
   - Encode categorical features (genre, director, actors)
   - Create target variable: Hit (Rating ≥ 6.5) vs Flop

2. **Feature Engineering:**
   - Year normalization
   - Duration analysis
   - Genre encoding using LabelEncoder
   - Director and actor popularity metrics

3. **Model Training:**
   - Algorithm: Random Forest Classifier
   - Estimators: 100 trees
   - Max Depth: 10 levels
   - Features: Year, Duration, Genre, Director, Lead Actor

### Recommendation System
1. **Content Analysis:**
   - Extract movie features (genre, director, actors)
   - Build movie profile vectors
   - Calculate similarity scores

2. **User Profiling:**
   - Track user ratings (Like/Dislike)
   - Build preference vector
   - Update with each new rating

3. **Recommendation Generation:**
   - Find movies with similar features to liked movies
   - Exclude already rated movies
   - Sort by similarity score
   - Return top matches

## ⚙️ Configuration

### Model Parameters (in movie.py)
python
# Random Forest parameters
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    random_state=42,       # Reproducibility
    max_depth=10,          # Tree depth
    min_samples_split=2,   # Minimum samples to split node
    min_samples_leaf=1     # Minimum samples in leaf node
)
### System Settings
- **Minimum ratings for recommendations**: 3
- **Default popular movies count**: 12
- **Search results limit**: 50
- **Port**: 5000 (configurable in app.py)

## 📱 UI Components

### Navigation
- **Movie Predictor Tab**: Prediction interface
- **Movie Recommender Tab**: Recommendation interface

### Prediction Interface
- Two input forms (by name / by details)
- Loading spinner during processing
- Results panel with animations

### Recommendation Interface
- Progress tracker
- Search box with genre filters
- Movie grid with rating buttons
- Recommendations display

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to load movies" error**

   Solution: Check if movies_dataset.csv exists in backend folder


2. **Flask server won't start**

   Solution: Check port 5000 availability or change port in app.py


3. **No predictions/recommendations**

   Solution: Verify dataset format and required columns


4. **CORS errors in browser console**

   Solution: Ensure Flask-CORS is properly installed and configured
   

### Debug Mode
Run Flask with debug mode for detailed errors:
bash
python app.py
# Server runs on http://localhost:5000 with debug enabled


## 📈 Performance Optimization

### For Large Datasets
1. **Dataset Optimization:**
   - Use efficient data types in pandas
   - Implement chunking for very large files
   - Add database support (SQLite/PostgreSQL)

2. **Model Optimization:**
   - Implement model persistence (joblib)
   - Add caching for frequent predictions
   - Use vectorized operations

3. **Frontend Optimization:**
   - Implement lazy loading for movie cards
   - Add pagination for search results
   - Minimize API calls with client-side caching

## 🔄 Extending the System

### Adding New Features
1. **Additional ML Models:**
   - Add regression for rating prediction
   - Implement collaborative filtering
   - Add sentiment analysis for reviews

2. **Enhanced UI Features:**
   - Add movie posters and trailers
   - Implement watchlists
   - Add social sharing features

3. **Data Sources:**
   - Integrate with IMDb API for real-time data
   - Add user review scraping
   - Include box office data

### Code Organization
- Keep business logic in `movie.py`
- API endpoints in `app.py`
- Frontend logic in `index.html` JavaScript
- Consider separating into modules for scalability

## 📝 Notes

### Development Notes
- The system uses in-memory storage for user sessions
- All ratings reset on server restart
- Sample data is generated if no dataset is found
- No authentication required (single-user mode)

### Deployment Considerations
- For production, add error logging
- Implement proper database for user data
- Add API rate limiting
- Secure CORS configuration
- Add input validation and sanitization

---

**Happy movie watching! 🍿**
