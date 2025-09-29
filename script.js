// Global variables
let model;
let isTraining = false;
let trainingProgress = 0;

// Initialize when window loads
window.onload = async function() {
    console.log('Initializing application...');
    
    try {
        // Load data first
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('result').innerHTML = 
            '<p style="color: red;">Error initializing application: ' + error.message + '</p>';
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Create options for users (using synthetic user IDs)
    for (let i = 1; i <= numUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = `${movie.title} (${movie.genres})`;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 10) {
    console.log(`Creating model with ${numUsers} users, ${numMovies} movies, latent dimension: ${latentDim}`);
    
    // Input layers for user and movie IDs
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // EMBEDDING LAYERS
    // Create embedding layers to learn latent factors for users and movies
    // These layers convert sparse user/movie IDs into dense vectors in latent space
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // LATENT VECTORS
    // Reshape embeddings to remove the extra dimension
    const userVector = tf.layers.flatten().apply(userEmbedding);
    const movieVector = tf.layers.flatten().apply(movieEmbedding);
    
    // PREDICTION
    // Compute dot product of user and movie vectors to get the predicted rating
    // This represents the interaction between user preferences and movie characteristics
    const dotProduct = tf.layers.dot({axes: 1}).apply([userVector, movieVector]);
    
    // Add bias terms for users and movies
    const userBias = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: 1,
        name: 'userBias'
    }).apply(userInput);
    
    const movieBias = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: 1,
        name: 'movieBias'
    }).apply(movieInput);
    
    const flattenedUserBias = tf.layers.flatten().apply(userBias);
    const flattenedMovieBias = tf.layers.flatten().apply(movieBias);
    
    // Combine dot product with bias terms
    const prediction = tf.layers.add().apply([
        dotProduct, 
        flattenedUserBias, 
        flattenedMovieBias
    ]);
    
    // Create and return the model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: prediction
    });
    
    return model;
}

async function trainModel() {
    if (isTraining) return;
    
    isTraining = true;
    const resultDiv = document.getElementById('result');
    const statusDiv = document.querySelector('.status-text');
    const progressFill = document.querySelector('.progress-fill');
    const predictBtn = document.getElementById('predict-btn');
    
    try {
        resultDiv.innerHTML = '<p>Starting model training... This may take a few moments.</p>';
        statusDiv.textContent = 'Creating model architecture...';
        
        // Create model
        model = createModel(numUsers, numMovies, 10);
        
        statusDiv.textContent = 'Compiling model...';
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        statusDiv.textContent = 'Preparing training data...';
        
        // Prepare training data
        const userData = [];
        const movieData = [];
        const ratingData = [];
        
        ratings.forEach(([userId, movieId, rating]) => {
            userData.push(userId);
            movieData.push(movieId);
            ratingData.push(rating);
        });
        
        const userTensor = tf.tensor2d(userData, [userData.length, 1]);
        const movieTensor = tf.tensor2d(movieData, [movieData.length, 1]);
        const ratingTensor = tf.tensor2d(ratingData, [ratingData.length, 1]);
        
        statusDiv.textContent = 'Training model...';
        
        // Train model
        const history = await model.fit(
            [userTensor, movieTensor],
            ratingTensor,
            {
                epochs: 8,
                batchSize: 32,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        trainingProgress = ((epoch + 1) / 8) * 100;
                        progressFill.style.width = trainingProgress + '%';
                        statusDiv.textContent = `Training epoch ${epoch + 1}/8 - Loss: ${logs.loss.toFixed(4)}`;
                        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}`);
                    }
                }
            }
        );
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        statusDiv.textContent = 'Training completed!';
        resultDiv.innerHTML = '<p>Model training completed successfully! You can now make predictions.</p>';
        predictBtn.disabled = false;
        
        console.log('Model training completed');
        
    } catch (error) {
        console.error('Training error:', error);
        resultDiv.innerHTML = '<p style="color: red;">Error training model: ' + error.message + '</p>';
        statusDiv.textContent = 'Training failed';
    } finally {
        isTraining = false;
    }
}

async function predictRating() {
    if (!model) {
        alert('Model is not ready yet. Please wait for training to complete.');
        return;
    }
    
    const userId = parseInt(document.getElementById('user-select').value);
    const movieId = parseInt(document.getElementById('movie-select').value);
    const resultDiv = document.getElementById('result');
    
    if (!userId || !movieId) {
        resultDiv.innerHTML = '<p style="color: red;">Please select both a user and a movie.</p>';
        return;
    }
    
    try {
        resultDiv.innerHTML = '<p>Calculating prediction...</p>';
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]], [1, 1]);
        const movieTensor = tf.tensor2d([[movieId]], [1, 1]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = rating[0];
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result
        const movie = movies.find(m => m.id === movieId);
        const clampedRating = Math.min(Math.max(predictedRating, 0.5), 5.0);
        
        // Create star rating display
        const fullStars = Math.round(clampedRating);
        const stars = '★'.repeat(fullStars) + '☆'.repeat(5 - fullStars);
        
        resultDiv.innerHTML = `
            <div class="rating-display">
                <p>Predicted rating for <strong>${movie.title}</strong> by <strong>User ${userId}</strong>:</p>
                <div class="stars">${stars}</div>
                <p><strong>${clampedRating.toFixed(1)} / 5.0</strong></p>
            </div>
        `;
        
        console.log(`Predicted rating: User ${userId}, Movie ${movieId} -> ${clampedRating.toFixed(1)}`);
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultDiv.innerHTML = '<p style="color: red;">Error making prediction: ' + error.message + '</p>';
    }
}

// Export functions for use in HTML
window.predictRating = predictRating;
