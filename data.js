// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// MovieLens dataset URLs
const MOVIES_URL = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat';
const RATINGS_URL = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat';

// Alternative MovieLens 100K dataset URLs (more reliable)
const ALT_MOVIES_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation/data/movie_names.csv';
const ALT_RATINGS_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation/data/ratings.csv';

async function loadData() {
    try {
        console.log('Loading movie data...');
        
        // For demonstration, we'll use synthetic data if external URLs fail
        await loadSyntheticData();
        
        console.log('Data loaded successfully');
        console.log(`Loaded ${movies.length} movies and ${ratings.length} ratings`);
        console.log(`Unique users: ${numUsers}, Unique movies: ${numMovies}`);
        
    } catch (error) {
        console.error('Error loading external data, using synthetic data:', error);
        await loadSyntheticData();
    }
}

function loadSyntheticData() {
    // Create synthetic movie data
    movies = [
        { id: 1, title: "The Shawshank Redemption", genres: "Drama" },
        { id: 2, title: "The Godfather", genres: "Crime|Drama" },
        { id: 3, title: "The Dark Knight", genres: "Action|Crime|Drama" },
        { id: 4, title: "Pulp Fiction", genres: "Crime|Drama" },
        { id: 5, title: "Forrest Gump", genres: "Drama|Romance" },
        { id: 6, title: "Inception", genres: "Action|Adventure|Sci-Fi" },
        { id: 7, title: "The Matrix", genres: "Action|Sci-Fi" },
        { id: 8, title: "Goodfellas", genres: "Biography|Crime|Drama" },
        { id: 9, title: "The Silence of the Lambs", genres: "Crime|Drama|Thriller" },
        { id: 10, title: "Star Wars: A New Hope", genres: "Action|Adventure|Fantasy" }
    ];

    // Create synthetic ratings data (user_id, movie_id, rating)
    ratings = [
        [1, 1, 5], [1, 2, 4], [1, 3, 5], [1, 6, 4],
        [2, 1, 4], [2, 4, 5], [2, 5, 4], [2, 7, 3],
        [3, 2, 5], [3, 3, 4], [3, 8, 5], [3, 10, 4],
        [4, 4, 4], [4, 5, 3], [4, 6, 5], [4, 9, 4],
        [5, 1, 5], [5, 7, 4], [5, 8, 3], [5, 10, 5],
        [6, 2, 4], [6, 3, 5], [6, 6, 4], [6, 9, 3],
        [7, 4, 5], [7, 5, 4], [7, 7, 5], [7, 8, 4],
        [8, 1, 3], [8, 2, 5], [8, 9, 4], [8, 10, 5]
    ];

    // Calculate unique users and movies
    const userSet = new Set();
    const movieSet = new Set();
    
    ratings.forEach(([userId, movieId, rating]) => {
        userSet.add(userId);
        movieSet.add(movieId);
    });
    
    numUsers = Math.max(...userSet) + 1; // +1 because IDs start from 1
    numMovies = Math.max(...movieSet) + 1;
    
    return Promise.resolve();
}

function parseItemData(text) {
    // Parse movie data from various formats
    const lines = text.split('\n');
    movies = [];
    
    lines.forEach(line => {
        if (line.trim()) {
            // Try different delimiter formats
            const parts = line.split(/[,|]/);
            if (parts.length >= 2) {
                const id = parseInt(parts[0]);
                const title = parts[1].trim();
                if (id && title) {
                    movies.push({ id, title, genres: parts[2] || 'Unknown' });
                }
            }
        }
    });
    
    return movies;
}

function parseRatingData(text) {
    // Parse rating data from various formats
    const lines = text.split('\n');
    ratings = [];
    const userSet = new Set();
    const movieSet = new Set();
    
    lines.forEach(line => {
        if (line.trim()) {
            const parts = line.split(/[,|]/);
            if (parts.length >= 3) {
                const userId = parseInt(parts[0]);
                const movieId = parseInt(parts[1]);
                const rating = parseFloat(parts[2]);
                
                if (userId && movieId && rating) {
                    ratings.push([userId, movieId, rating]);
                    userSet.add(userId);
                    movieSet.add(movieId);
                }
            }
        }
    });
    
    numUsers = Math.max(...userSet) + 1;
    numMovies = Math.max(...movieSet) + 1;
    
    return ratings;
}

// Export functions for use in script.js
window.loadData = loadData;
window.parseItemData = parseItemData;
window.parseRatingData = parseRatingData;
