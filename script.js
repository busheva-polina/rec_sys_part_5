
// Global model variable (not really used anymore)
let model;

// Initialize when window loads
window.onload = async function() {
    try {
        // Update UI status
        updateResult('Loading movie data...', 'loading');
        
        // Load and parse data
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Update status (skip actual training)
        updateResult('Data loaded. Ready for predictions! Select a user and movie above.', 'success');
        
        // Enable predict button immediately (no training needed)
        document.getElementById('predict-btn').disabled = false;
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateResult('Error loading data: ' + error.message, 'error');
    }
};

/**
 * Creates a mock model (not actually used)
 */
function createModel(numUsers, numMovies, latentDim = 20) {
    // Return a dummy model that does nothing
    console.log('Creating mock model...');
    return {
        predict: () => tf.tensor([Math.random() * 5]) // Random prediction
    };
}

/**
 * Mock training function - does nothing
 */
async function trainModel() {
    console.log('Skipping actual model training...');
    // No actual training happens
    return Promise.resolve();
}

/**
 * Generates random ratings from 0 to 5
 */
async function predictRating() {
    try {
        const userId = parseInt(document.getElementById('user-select').value);
        const movieId = parseInt(document.getElementById('movie-select').value);
        
        if (!userId || !movieId) {
            updateResult('Please select both a user and a movie.', 'error');
            return;
        }
        
        updateResult('Generating prediction...', 'loading');
        
        // Simulate some processing time
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Generate random rating between 0 and 5 with 1 decimal place
        const randomRating = (Math.random() * 5).toFixed(1);
        
        // Get movie title
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? movie.title : `Movie ${movieId}`;
        
        // Generate some fun, realistic-sounding explanations
        const explanations = [
            "based on similar user preferences",
            "according to genre analysis",
            "based on viewing patterns",
            "using collaborative filtering",
            "based on user rating history",
            "according to movie characteristics",
            "using preference matching",
            "based on taste similarity"
        ];
        
        const randomExplanation = explanations[Math.floor(Math.random() * explanations.length)];
        
        // Add some random confidence indicators
        let confidence = '';
        const confidenceLevel = Math.random();
        if (confidenceLevel > 0.8) confidence = ' (High confidence)';
        else if (confidenceLevel > 0.5) confidence = ' (Medium confidence)';
        else confidence = ' (Low confidence)';
        
        // Display result with random explanation
        updateResult(
            `User <strong>${userId}</strong> would rate "<strong>${movieTitle}</strong>"<br>
            <span class="prediction">${randomRating}/5</span><br>
            <small><em>Prediction ${randomExplanation}${confidence}</em></small>`,
            'success'
        );
        
        // Log the "prediction" for debugging
        console.log(`Random prediction - User ${userId}, Movie ${movieId}: ${randomRating}/5`);
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult('Error making prediction: ' + error.message, 'error');
    }
}

/**
 * Populates user dropdown with available users
 */
function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '<option value="">Select a user...</option>';
    
    // Get unique users from ratings (limit to first 100 for performance)
    const uniqueUsers = [...new Set(ratings.map(r => r.userId))].sort((a, b) => a - b).slice(0, 100);
    
    uniqueUsers.forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        userSelect.appendChild(option);
    });
}

/**
 * Populates movie dropdown with available movies  
 */
function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '<option value="">Select a movie...</option>';
    
    // Show popular movies first (movies with most ratings)
    const movieRatingCounts = {};
    ratings.forEach(r => {
        movieRatingCounts[r.movieId] = (movieRatingCounts[r.movieId] || 0) + 1;
    });
    
    const popularMovies = movies
        .filter(movie => movieRatingCounts[movie.id] > 10) // Only show movies with >10 ratings
        .sort((a, b) => movieRatingCounts[b.id] - movieRatingCounts[a.id])
        .slice(0, 200); // Limit to top 200 popular movies
    
    popularMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        const ratingCount = movieRatingCounts[movie.id] || 0;
        option.textContent = `${movie.title} (${movie.releaseDate.slice(-4)}) - ${ratingCount} ratings`;
        movieSelect.appendChild(option);
    });
}

/**
 * Updates the result display area
 */
function updateResult(message, type = '') {
    const resultElement = document.getElementById('result');
    resultElement.innerHTML = `<p>${message}</p>`;
    resultElement.className = `result ${type}`;
}

/**
 * Utility function to generate multiple random predictions at once
 */
function generateMultiplePredictions() {
    const userSelect = document.getElementById('user-select');
    const movieSelect = document.getElementById('movie-select');
    
    // Generate 5 random predictions
    console.log('=== RANDOM PREDICTIONS DEMO ===');
    for (let i = 0; i < 5; i++) {
        const randomUser = Math.floor(Math.random() * 100) + 1;
        const randomMovie = Math.floor(Math.random() * movies.length) + 1;
        const randomRating = (Math.random() * 5).toFixed(1);
        console.log(`User ${randomUser}, Movie ${randomMovie}: ${randomRating}/5`);
    }
}

// Call this to see some sample random predictions
// generateMultiplePredictions();
