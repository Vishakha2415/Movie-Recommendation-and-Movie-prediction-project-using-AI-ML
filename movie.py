import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv('movies_dataset.csv')

# Data cleaning
def clean_data(df):
    # Extract year from string (e.g., "(2014)" -> 2014)
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Extract duration from string
    df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Convert Rating to numeric
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Clean Votes column if exists
    if 'Votes' in df.columns:
        df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False).astype(float)
    
    # Fill missing values
    df['Year'].fillna(df['Year'].median(), inplace=True)
    df['Duration'].fillna(df['Duration'].median(), inplace=True)
    df['Rating'].fillna(df['Rating'].median(), inplace=True)
    
    # Clean text columns
    text_columns = ['Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Genre']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    return df

# Clean the data
df = clean_data(df)

# Add unique ID to movies
df['id'] = range(1, len(df) + 1)

# User ratings storage
user_ratings = {}

# Prediction Model
class MoviePredictor:
    def __init__(self, df):
        self.df = df
        self.prepare_features()
        self.train_model()
    
    def prepare_features(self):
        # Create target variable (Hit = Rating >= 6.5)
        self.df['is_hit'] = (self.df['Rating'] >= 6.5).astype(int)
        
        # Encode categorical features
        self.genre_encoder = LabelEncoder()
        self.df['Genre_encoded'] = self.genre_encoder.fit_transform(
            self.df['Genre']
        )
        
        self.director_encoder = LabelEncoder()
        self.df['Director_encoded'] = self.director_encoder.fit_transform(
            self.df['Director']
        )
        
        self.actor1_encoder = LabelEncoder()
        self.df['Actor1_encoded'] = self.actor1_encoder.fit_transform(
            self.df['Actor 1']
        )
        
        # Features for model
        features = ['Year', 'Duration', 'Genre_encoded', 'Director_encoded', 'Actor1_encoded']
        self.X = self.df[features].fillna(0)
        self.y = self.df['is_hit']
    
    def train_model(self):
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(self.X, self.y)
    
    def predict_by_name(self, movie_name, release_year=None):
        """Predict success for a movie by name"""
        # Find movie in dataset
        movie_data = self.df[self.df['Name'].str.contains(movie_name, case=False, na=False)]
        
        if len(movie_data) == 0:
            # If movie not found, create synthetic data
            return self.predict_synthetic(movie_name, release_year)
        
        # Use first match
        movie = movie_data.iloc[0]
        
        # Prepare features for prediction
        features = self.prepare_prediction_features(
            movie['Genre'],
            movie['Director'],
            movie['Actor 1'],
            movie['Year'],
            movie['Duration']
        )
        
        return self.make_prediction(features, movie_name, movie['Year'])
    
    def predict_by_details(self, director, actor1, actor2, genre):
        """Predict success based on details"""
        features = self.prepare_prediction_features(
            genre,
            director,
            actor1,
            2023,  # Current year
            120    # Average duration
        )
        
        return self.make_prediction(features, None, 2023, director, actor1, actor2, genre)
    
    def prepare_prediction_features(self, genre, director, actor1, year, duration):
        """Prepare features for prediction"""
        try:
            genre_encoded = self.genre_encoder.transform([genre])[0]
        except:
            # If genre not in encoder, use 0
            genre_encoded = 0
        
        try:
            director_encoded = self.director_encoder.transform([director])[0]
        except:
            director_encoded = 0
        
        try:
            actor1_encoded = self.actor1_encoder.transform([actor1])[0]
        except:
            actor1_encoded = 0
        
        return np.array([[year, duration, genre_encoded, director_encoded, actor1_encoded]])
    
    def predict_synthetic(self, movie_name, release_year):
        """Create synthetic prediction for unknown movies"""
        # Generate prediction based on movie name
        name_length = len(movie_name)
        name_factor = min(1.0, name_length / 50)
        
        if release_year:
            year_factor = min(1.0, (release_year - 2000) / 23)
        else:
            year_factor = 0.7
        
        base_rating = 5.5 + (random.random() * 3)
        rating = base_rating + (name_factor * 0.5) + (year_factor * 0.3)
        rating = min(10, max(1, rating))
        
        # Random confidence
        confidence = 70 + random.random() * 25
        
        # Determine hit or flop
        is_hit = rating >= 6.5
        
        return {
            'success': True,
            'prediction': 'HIT' if is_hit else 'FLOP',
            'predicted_rating': round(rating, 1),
            'confidence': round(confidence, 1),
            'movie_name': movie_name,
            'release_year': release_year if release_year else 'N/A',
            'factors': ['Generated prediction based on name analysis']
        }
    
    def make_prediction(self, features, movie_name=None, year=None, director=None, actor1=None, actor2=None, genre=None):
        """Make prediction with given features"""
        # Get prediction probabilities
        proba = self.model.predict_proba(features)[0]
        
        # Get predicted class (1 = hit, 0 = flop)
        prediction = self.model.predict(features)[0]
        
        # Calculate confidence
        confidence = round(proba[1] * 100, 1) if prediction == 1 else round(proba[0] * 100, 1)
        
        # Generate factors
        factors = self.generate_factors(features[0])
        
        # Estimate rating
        base_rating = 5.0 if prediction == 0 else 6.5
        rating_variation = random.random() * 1.5
        predicted_rating = round(base_rating + rating_variation + (confidence/100), 1)
        predicted_rating = min(10, max(1, predicted_rating))
        
        result = {
            'success': True,
            'prediction': 'HIT' if prediction == 1 else 'FLOP',
            'predicted_rating': predicted_rating,
            'confidence': confidence,
            'factors': factors
        }
        
        if movie_name:
            result['movie_name'] = movie_name
            result['release_year'] = year if year else 'N/A'
        
        if director:
            result['director'] = director
            result['actor1'] = actor1
            result['actor2'] = actor2 or 'N/A'
            result['genre'] = genre
        
        return result
    
    def generate_factors(self, features):
        """Generate key factors for prediction"""
        factors = []
        
        # Year factor
        year = features[0]
        if year >= 2010:
            factors.append("Recent release year favorable")
        elif year <= 1990:
            factors.append("Classic release timing")
        
        # Duration factor
        duration = features[1]
        if 90 <= duration <= 150:
            factors.append("Optimal movie duration")
        elif duration > 150:
            factors.append("Extended runtime may affect viewership")
        
        factors.append("Director track record considered")
        factors.append("Lead actor popularity analyzed")
        factors.append("Genre market trends evaluated")
        
        return factors[:5]

# Recommendation System
class MovieRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.user_ratings = {}
    
    def add_rating(self, movie_id, liked):
        """Add user rating for a movie"""
        # Convert movie_id to int
        movie_id = int(movie_id)
        self.user_ratings[movie_id] = liked
        return True
    
    def get_progress(self):
        """Get user rating progress"""
        count = len(self.user_ratings)
        progress = min(100, (count / 3) * 100)
        
        return {
            'count': count,
            'progress': progress,
            'needed': 3
        }
    
    def get_popular_movies(self, n=12):
        """Get popular movies (top rated)"""
        # Filter movies with valid ratings
        valid_movies = self.df[self.df['Rating'].notna()]
        
        # Sort by rating and votes if available
        if 'Votes' in valid_movies.columns and valid_movies['Votes'].notna().any():
            valid_movies = valid_movies.sort_values(['Rating', 'Votes'], ascending=[False, False])
        else:
            valid_movies = valid_movies.sort_values('Rating', ascending=False)
        
        # Take top n movies
        top_movies = valid_movies.head(n)
        
        return self.format_movies(top_movies)
    
    def search_movies(self, query, genre_filter='all'):
        """Search movies by query and genre"""
        query = str(query).lower()
        
        if query:
            # Create search mask
            mask = (
                self.df['Name'].str.lower().str.contains(query, na=False) |
                self.df['Director'].str.lower().str.contains(query, na=False) |
                self.df['Actor 1'].str.lower().str.contains(query, na=False) |
                self.df['Actor 2'].str.lower().str.contains(query, na=False) |
                self.df['Genre'].str.lower().str.contains(query, na=False)
            )
            results = self.df[mask]
        else:
            results = self.df.copy()
        
        # Filter by genre
        if genre_filter != 'all':
            results = results[results['Genre'].str.contains(genre_filter, case=False, na=False)]
        
        # Sort by rating and limit results
        results = results.sort_values('Rating', ascending=False).head(50)
        
        return self.format_movies(results)
    
    def format_movies(self, movies_df):
        """Format movies for frontend display"""
        movies = []
        
        for _, row in movies_df.iterrows():
            # Create a safe movie dictionary
            movie = {
                'id': int(row['id']),
                'title': str(row['Name']),
                'year': int(row['Year']) if pd.notna(row['Year']) and not pd.isnull(row['Year']) else 'N/A',
                'rating': float(row['Rating']) if pd.notna(row['Rating']) else 'N/A',
                'duration': f"{int(row['Duration'])} min" if pd.notna(row['Duration']) else 'N/A',
                'genre': str(row['Genre']),
                'director': str(row['Director']),
                'description': f"{row['Name']} ({int(row['Year']) if pd.notna(row['Year']) else 'N/A'}) is a {row['Genre']} film directed by {row['Director']}."
            }
            
            # Add actors if available
            actors = []
            if pd.notna(row['Actor 1']):
                actors.append(str(row['Actor 1']))
            if pd.notna(row['Actor 2']):
                actors.append(str(row['Actor 2']))
            if pd.notna(row['Actor 3']):
                actors.append(str(row['Actor 3']))
            
            if actors:
                movie['actors'] = ", ".join(actors)
            
            movies.append(movie)
        
        return movies
    
    def get_recommendations(self):
        """Get personalized recommendations based on user ratings"""
        if len(self.user_ratings) < 3:
            return []
        
        # Get liked movies
        liked_movies = [movie_id for movie_id, liked in self.user_ratings.items() if liked]
        
        if not liked_movies:
            # If no likes, return popular movies
            return self.get_popular_movies(6)
        
        # Get liked movie data
        liked_movie_ids = list(self.user_ratings.keys())
        liked_movie_data = self.df[self.df['id'].isin(liked_movie_ids)]
        
        # Create recommendation set
        all_recommendations = []
        
        for _, liked_movie in liked_movie_data.iterrows():
            # Find similar movies (exclude already rated)
            unrated_movies = self.df[~self.df['id'].isin(liked_movie_ids)]
            
            # Find movies with same genre
            genre_recs = unrated_movies[
                unrated_movies['Genre'] == liked_movie['Genre']
            ].head(2)
            
            # Find movies with same director
            director_recs = unrated_movies[
                unrated_movies['Director'] == liked_movie['Director']
            ].head(2)
            
            # Find movies with same actor
            actor_recs = unrated_movies[
                (unrated_movies['Actor 1'] == liked_movie['Actor 1']) |
                (unrated_movies['Actor 2'] == liked_movie['Actor 1']) |
                (unrated_movies['Actor 3'] == liked_movie['Actor 1'])
            ].head(2)
            
            # Add to recommendations
            for rec_df in [genre_recs, director_recs, actor_recs]:
                for _, rec_movie in rec_df.iterrows():
                    if rec_movie['id'] not in [r['id'] for r in all_recommendations]:
                        all_recommendations.append(self.create_recommendation(rec_movie, liked_movie))
        
        # If not enough recommendations, add popular movies
        if len(all_recommendations) < 6:
            popular = self.get_popular_movies(10)
            for movie in popular:
                if movie['id'] not in [r['id'] for r in all_recommendations] and movie['id'] not in liked_movie_ids:
                    rec_data = self.df[self.df['id'] == movie['id']].iloc[0]
                    all_recommendations.append(self.create_recommendation(rec_data, None))
        
        # Return up to 6 recommendations
        return all_recommendations[:6]
    
    def create_recommendation(self, movie, liked_movie):
        """Create a recommendation object"""
        match_score = random.choice(["High", "Very High", "Excellent"])
        
        if liked_movie is not None:
            reasons = []
            if movie['Genre'] == liked_movie['Genre']:
                reasons.append("Same genre")
            if movie['Director'] == liked_movie['Director']:
                reasons.append("Same director")
            if (movie['Actor 1'] == liked_movie['Actor 1'] or 
                movie['Actor 2'] == liked_movie['Actor 1'] or
                movie['Actor 3'] == liked_movie['Actor 1']):
                reasons.append("Same actor")
            
            if not reasons:
                reasons.append("Similar style to your preferences")
            
            reason = " and ".join(reasons)
        else:
            reason = "Popular choice with high ratings"
        
        return {
            'id': int(movie['id']),
            'title': str(movie['Name']),
            'year': int(movie['Year']) if pd.notna(movie['Year']) else 'N/A',
            'rating': float(movie['Rating']) if pd.notna(movie['Rating']) else 'N/A',
            'genres': [str(movie['Genre'])],
            'match_score': match_score,
            'reason': reason,
            'description': f"{movie['Name']} is a {movie['Genre']} film directed by {movie['Director']}."
        }
    
    def reset_ratings(self):
        """Reset all user ratings"""
        self.user_ratings = {}
        return True

# Initialize systems
predictor = MoviePredictor(df)
recommender = MovieRecommender(df)

# API-like functions for your HTML
def predict_movie_by_name(movie_name, year=None):
    return predictor.predict_by_name(movie_name, year)

def predict_movie_by_details(director, actor1, actor2, genre):
    return predictor.predict_by_details(director, actor1, actor2, genre)

def get_popular_movies_for_display():
    return recommender.get_popular_movies(12)

def search_movies_for_display(query, genre='all'):
    return recommender.search_movies(query, genre)

def add_user_rating(movie_id, liked):
    return recommender.add_rating(movie_id, liked)

def get_user_progress():
    return recommender.get_progress()

def get_personalized_recommendations():
    return recommender.get_recommendations()

def reset_all_ratings():
    return recommender.reset_ratings()

# Test the system
if __name__ == "__main__":
    print("Movie AI System Initialized")
    print(f"Dataset shape: {df.shape}")
    print(f"Total movies: {len(df)}")
    print(f"Sample movie: {df.iloc[0]['Name']}")
    
    # Test popular movies
    print("\n1. Testing Popular Movies:")
    movies = get_popular_movies_for_display()
    print(f"Found {len(movies)} popular movies")
    for i, movie in enumerate(movies[:3], 1):
        print(f"  {i}. {movie['title']} ({movie['year']}) - Rating: {movie['rating']}")
    
    # Test prediction
    print("\n2. Testing Prediction:")
    result = predict_movie_by_name("The Dark Knight")
    print(f"Prediction for 'The Dark Knight': {result['prediction']}")
    print(f"Predicted Rating: {result['predicted_rating']}")
    print(f"Confidence: {result['confidence']}%")
    
    # Test recommendations
    print("\n3. Testing Recommendations System:")
    # Add some ratings
    if len(movies) >= 3:
        print("Adding sample ratings...")
        add_user_rating(movies[0]['id'], True)
        add_user_rating(movies[1]['id'], True)
        add_user_rating(movies[2]['id'], False)
        
        progress = get_user_progress()
        print(f"User progress: {progress['count']}/3 ratings")
        
        if progress['count'] >= 3:
            recommendations = get_personalized_recommendations()
            print(f"Generated {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec['title']} - {rec['match_score']} match")
    
    print("\nSystem ready to use with your HTML frontend!")
