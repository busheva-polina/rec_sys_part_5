// Global variables
let model;
let isTraining = false;

// Initialize when window loads
window.onload = async function() {
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
            '<div class="error">Error loading data or training model. Please check console for details.</div>';
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Add users (using first 1000 users for performance)
    const maxUsers = Math.min(1000, numUsers);
    for (let i = 1; i <= maxUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Add movies
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        const displayText = movie.year ? `${movie.title} (${movie.year})` : movie.title;
        option.textContent = displayText;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 10) {
    // Input layers
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Embedding layers
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
    
    // Flatten embeddings
    const userFlatten = tf.layers.flatten().apply(userEmbedding);
    const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
    
    // Dot product of user and movie embeddings
    const dotProduct = tf.layers.dot({axes: -1}).apply([userFlatten, movieFlatten]);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: dotProduct
    });
    
    console.log('Model created successfully');
    return model;
}

async function trainModel() {
    try {
        isTraining = true;
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<div class="loading">Training model... This may take a few minutes.</div>';
        
        // Create model
        model = createModel(numUsers, numMovies, 10);
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        // Prepare training data
        const userIds = ratings.map(r => r.userId);
        const movieIds = ratings.map(r => r.movieId);
        const ratingValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        console.log('Starting model training...');
        
        // Train model
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 5,
            batchSize: 64,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    resultDiv.innerHTML = 
                        `<div class="loading">Training model... Epoch ${epoch + 1}/5, Loss: ${logs.loss.toFixed(4)}</div>`;
                }
            }
        });
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        ratingTensor.dispose();
        
        // Enable predict button
        document.getElementById('predict-btn').disabled = false;
        isTraining = false;
        
        resultDiv.innerHTML = '<div class="success">Model training completed! You can now make predictions.</div>';
        console.log('Model training completed');
        
    } catch (error) {
        console.error('Training error:', error);
        document.getElementById('result').innerHTML = 
            '<div class="error">Error training model. Please check console for details.</div>';
        isTraining = false;
    }
}

async function predictRating() {
    if (isTraining) {
        alert('Model is still training. Please wait...');
        return;
    }
    
    const userId = parseInt(document.getElementById('user-select').value);
    const movieId = parseInt(document.getElementById('movie-select').value);
    const resultDiv = document.getElementById('result');
    
    if (!userId || !movieId) {
        resultDiv.innerHTML = '<div class="error">Please select both a user and a movie.</div>';
        return;
    }
    
    try {
        resultDiv.innerHTML = '<div class="loading">Making prediction...</div>';
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]]);
        const movieTensor = tf.tensor2d([[movieId]]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = rating[0];
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        prediction.dispose();
        
        // Get movie title
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;
        
        // Display result
        const clampedRating = Math.min(Math.max(predictedRating, 1), 5); // Clamp between 1-5
        resultDiv.innerHTML = `
            <div class="success">
                <strong>Predicted Rating:</strong><br>
                User ${userId} would rate "${movieTitle}"<br>
                <span style="font-size: 1.5em; color: #ff6b35;">${clampedRating.toFixed(1)} / 5 stars</span>
            </div>
        `;
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultDiv.innerHTML = '<div class="error">Error making prediction. Please try again.</div>';
    }
}
