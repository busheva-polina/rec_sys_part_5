// Global model variable
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
        
        // Update status and start training
        updateResult('Data loaded. Training model... This may take a few moments.', 'loading');
        
        // Train the model
        await trainModel();
        
        // Enable predict button and update status
        document.getElementById('predict-btn').disabled = false;
        updateResult('Model trained and ready for predictions! Select a user and movie above.', 'success');
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateResult('Error initializing application: ' + error.message, 'error');
    }
};

/**
 * Creates the Matrix Factorization model architecture
 */
function createModel(numUsers, numMovies, latentDim = 10) {
    // Input Layers: Define separate inputs for user and movie IDs
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Embedding Layers: 
    // - Convert sparse user/movie indices into dense vectors of fixed size
    // - Each user/movie gets a unique latent vector representation
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers + 1,  // +1 because user IDs start at 1
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies + 1, // +1 because movie IDs start at 1  
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // Latent Vectors: 
    // - The output from embedding layers are the latent feature vectors
    // - userLatentVector: dense vector representing user's preferences
    // - movieLatentVector: dense vector representing movie's characteristics
    const userLatentVector = tf.layers.flatten().apply(userEmbedding);
    const movieLatentVector = tf.layers.flatten().apply(movieEmbedding);
    
    // Prediction: 
    // - Compute dot product of user and movie latent vectors
    // - Measures alignment between user preferences and movie features
    // - Higher dot product = higher predicted rating
    const dotProduct = tf.layers.dot({axes: -1}).apply([userLatentVector, movieLatentVector]);
    
    // Create and return the model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: dotProduct
    });
    
    return model;
}

/**
 * Trains the Matrix Factorization model
 */
async function trainModel() {
    try {
        // Create model with appropriate dimensions
        model = createModel(numUsers, numMovies, 10);
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        // Prepare training data
        const userIds = ratings.map(r => r.userId);
        const movieIds = ratings.map(r => r.movieId);
        const ratingsValues = ratings.map(r => r.rating);
        
        // Convert to tensors
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingsValues, [ratingsValues.length, 1]);
        
        // Train the model
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 8,
            batchSize: 64,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    updateResult(`Training... Epoch ${epoch + 1}/8 completed. Loss: ${logs.loss.toFixed(4)}`, 'loading');
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
    } catch (error) {
        console.error('Training error:', error);
        throw error;
    }
}

/**
 * Predicts rating for selected user and movie
 */
async function predictRating() {
    try {
        const userId = parseInt(document.getElementById('user-select').value);
        const movieId = parseInt(document.getElementById('movie-select').value);
        
        if (!userId || !movieId) {
            updateResult('Please select both a user and a movie.', 'error');
            return;
        }
        
        updateResult('Making prediction...', 'loading');
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]]);
        const movieTensor = tf.tensor2d([[movieId]]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        
        // Get movie title
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? movie.title : `Movie ${movieId}`;
        
        // Display result
        const predictedRating = rating[0].toFixed(1);
        updateResult(
            `Predicted rating for User ${userId} and "${movieTitle}": <span class="prediction">${predictedRating}/5</span>`,
            'success'
        );
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
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
    
    // Get unique users from ratings
    const uniqueUsers = [...new Set(ratings.map(r => r.userId))].sort((a, b) => a - b);
    
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
    
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = `${movie.title} (${movie.releaseDate.slice(-4)})`;
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
