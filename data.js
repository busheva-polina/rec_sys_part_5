// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// Movie data URL (MovieLens 100K dataset - u.item file)
const MOVIES_URL = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat';
// Ratings data URL (MovieLens 100K dataset - u.data file)
const RATINGS_URL = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat';

async function loadData() {
    try {
        console.log('Loading movie data...');
        const moviesResponse = await fetch(MOVIES_URL);
        const moviesText = await moviesResponse.text();
        movies = parseItemData(moviesText);
        
        console.log('Loading rating data...');
        const ratingsResponse = await fetch(RATINGS_URL);
        const ratingsText = await ratingsResponse.text();
        ratings = parseRatingData(ratingsText);
        
        console.log('Data loading completed');
        console.log(`Loaded ${movies.length} movies and ${ratings.length} ratings`);
        
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

function parseItemData(text) {
    const movies = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim()) {
            const parts = line.split('::');
            if (parts.length >= 2) {
                const movieId = parseInt(parts[0]);
                const titleParts = parts[1].split(' (');
                let title = parts[1];
                let year = null;
                
                // Extract year from title if present
                if (titleParts.length > 1) {
                    title = titleParts[0];
                    const yearMatch = titleParts[1].match(/(\d{4})/);
                    if (yearMatch) {
                        year = parseInt(yearMatch[1]);
                    }
                }
                
                movies.push({
                    id: movieId,
                    title: title,
                    year: year
                });
            }
        }
    }
    
    // Update global numMovies
    numMovies = movies.length;
    console.log(`Parsed ${numMovies} movies`);
    
    return movies;
}

function parseRatingData(text) {
    const ratings = [];
    const userSet = new Set();
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim()) {
            const parts = line.split('::');
            if (parts.length >= 3) {
                const userId = parseInt(parts[0]);
                const movieId = parseInt(parts[1]);
                const rating = parseFloat(parts[2]);
                
                ratings.push({
                    userId: userId,
                    movieId: movieId,
                    rating: rating
                });
                
                userSet.add(userId);
            }
        }
    }
    
    // Update global numUsers
    numUsers = Math.max(...userSet) + 1; // +1 because user IDs start from 1
    console.log(`Parsed ${ratings.length} ratings from ${numUsers} users`);
    
    return ratings;
}
