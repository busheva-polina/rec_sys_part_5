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
 * Creates the improved Matrix Factorization model architecture
 */
function createModel(numUsers, numMovies, latentDim = 20) {
    // Input Layers
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Improved Embedding Layers with better initialization
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers + 1,
        outputDim: latentDim,
        embeddingsInitializer: 'glorotNormal', // Better initialization
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies + 1,
        outputDim: latentDim,
        embeddingsInitializer: 'glorotNormal', // Better initialization
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // Latent Vectors with regularization
    const userLatentVector = tf.layers.flatten().apply(userEmbedding);
    const movieLatentVector = tf.layers.flatten().apply(movieEmbedding);
    
    // Add bias terms for users and movies
    const userBias = tf.layers.embedding({
        inputDim: numUsers + 1,
        outputDim: 1,
        embeddingsInitializer: 'zeros',
        name: 'userBias'
    }).apply(userInput);
    
    const movieBias = tf.layers.embedding({
        inputDim: numMovies + 1,
        outputDim: 1,
        embeddingsInitializer: 'zeros',
        name: 'movieBias'
    }).apply(movieInput);
    
    const userBiasFlatten = tf.layers.flatten().apply(userBias);
    const movieBiasFlatten = tf.layers.flatten().apply(movieBias);
    
    // Dot product with bias terms
    const dotProduct = tf.layers.dot({axes: -1}).apply([userLatentVector, movieLatentVector]);
    
    // Add biases to dot product
    const dotWithUserBias = tf.layers.add().apply([dotProduct, userBiasFlatten]);
    const prediction = tf.layers.add().apply([dotWithUserBias, movieBiasFlatten]);
    
    // Constrain output to rating range (1-5) using a custom activation
    const constrainedOutput = tf.layers.lambda({
        function: (x) => {
            // Scale and shift to approximate 1-5 range
            return x.sigmoid().mul(4).add(1);
        }
    }).apply(prediction);
    
    // Create and return the model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: constrainedOutput
    });
    
    return model;
}

/**
 * Improved training function with better parameters
 */
async function trainModel() {
    try {
        // Calculate global mean rating for normalization
        const globalMean = ratings.reduce((sum, r) => sum + r.rating, 0) / ratings.length;
        console.log(`Global mean rating: ${globalMean.toFixed(2)}`);
        
        // Create model with larger latent dimension
        model = createModel(numUsers, numMovies, 32); // Increased from 10 to 32
        
        // Compile the model with better learning rate
        model.compile({
            optimizer: tf.train.adam(0.01), // Increased learning rate
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
        
        // Print model summary
        model.summary();
        
        // Prepare training data with shuffling
        const shuffledRatings = [...ratings].sort(() => Math.random() - 0.5);
        
        const userIds = shuffledRatings.map(r => r.userId);
        const movieIds = shuffledRatings.map(r => r.movieId);
        const ratingsValues = shuffledRatings.map(r => r.rating);
        
        // Convert to tensors
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingsValues, [ratingsValues.length, 1]);
        
        // Train the model with more epochs and callbacks
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 20, // Increased epochs
            batchSize: 128, // Increased batch size
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}`);
                    const status = `Training... Epoch ${epoch + 1}/20 - Loss: ${logs.loss.toFixed(4)}${logs.val_loss ? `, Val Loss: ${logs.val_loss.toFixed(4)}` : ''}`;
                    updateResult(status, 'loading');
                    
                    // Early stopping check
                    if (logs.loss < 0.5) { // If loss is reasonable
                        document.getElementById('predict-btn').disabled = false;
                    }
                },
                onTrainEnd: () => {
                    console.log('Training completed');
                }
            }
        });
        
        // Test the model on a few samples
        await testModelSamples();
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
    } catch (error) {
        console.error('Training error:', error);
        throw error;
    }
}

/**
 * Test the model on some sample ratings to verify it's working
 */
async function testModelSamples() {
    console.log('Testing model on sample ratings...');
    
    // Test on first 5 ratings
    const testSamples = ratings.slice(0, 5);
    
    for (const sample of testSamples) {
        const userTensor = tf.tensor2d([[sample.userId]]);
        const movieTensor = tf.tensor2d([[sample.movieId]]);
        
        const prediction = model.predict([userTensor, movieTensor]);
        const predRating = (await prediction.data())[0];
        
        console.log(`User ${sample.userId}, Movie ${sample.movieId}: Actual=${sample.rating}, Predicted=${predRating.toFixed(2)}`);
        
        tf.dispose([userTensor, movieTensor, prediction]);
    }
}

/**
 * Improved prediction function with better error handling
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
        
        // Display result with confidence indication
        const predictedRating = rating[0];
        let confidence = '';
        
        if (predictedRating >= 4.5) confidence = ' (High rating expected)';
        else if (predictedRating >= 3.5) confidence = ' (Good match)';
        else if (predictedRating >= 2.5) confidence = ' (Neutral)';
        else confidence = ' (Low rating expected)';
        
        updateResult(
            `User ${userId} would rate "<strong>${movieTitle}</strong>"<br>
            <span class="prediction">${predictedRating.toFixed(1)}/5</span>${confidence}`,
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
 * Utility function to get model information
 */
function getModelInfo() {
    if (!model) return 'Model not yet trained';
    
    const trainableParams = model.trainableWeights.reduce((total, weight) => {
        return total + weight.shape.reduce((a, b) => a * b, 1);
    }, 0);
    
    return `Model: ${trainableParams.toLocaleString()} trainable parameters`;
}
