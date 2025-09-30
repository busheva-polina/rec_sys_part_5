// Global variables to store parsed data and dimensions
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// Movie data URL (u.item)
const MOVIES_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation-matrix-factorization/data/u.item';
// Ratings data URL (u.data)  
const RATINGS_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation-matrix-factorization/data/u.data';

/**
 * Loads and parses movie and rating data
 */
async function loadData() {
    try {
        // Load movie data
        const moviesResponse = await fetch(MOVIES_URL);
        const moviesText = await moviesResponse.text();
        movies = parseItemData(moviesText);
        numMovies = movies.length;

        // Load rating data
        const ratingsResponse = await fetch(RATINGS_URL);
        const ratingsText = await ratingsResponse.text();
        ratings = parseRatingData(ratingsText);
        
        // Calculate number of unique users
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        numUsers = uniqueUsers.size;

        console.log(`Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

/**
 * Parses movie item data from u.item file
 * Format: movieId|movieTitle|releaseDate|... 
 */
function parseItemData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    return lines.map(line => {
        const parts = line.split('|');
        return {
            id: parseInt(parts[0]),
            title: parts[1],
            releaseDate: parts[2],
            // Additional fields can be parsed as needed
        };
    });
}

/**
 * Parses rating data from u.data file
 * Format: userId|movieId|rating|timestamp
 */
function parseRatingData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    return lines.map(line => {
        const parts = line.split('\t');
        return {
            userId: parseInt(parts[0]),
            movieId: parseInt(parts[1]),
            rating: parseFloat(parts[2]),
            timestamp: parseInt(parts[3])
        };
    });
}
