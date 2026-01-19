// Client-side JavaScript for Outfit Extractor

let config = {
    server_url: "http://127.0.0.1:8000"
};

let selectedFile = null;

// Load configuration from localStorage or config.json
function loadConfig() {
    const savedConfig = localStorage.getItem('outfitExtractorConfig');
    if (savedConfig) {
        try {
            config = JSON.parse(savedConfig);
            document.getElementById('serverUrl').value = config.server_url;
        } catch (e) {
            console.error('Failed to parse saved config:', e);
        }
    } else {
        // Try to load from config.json
        fetch('config.json')
            .then(response => response.json())
            .then(data => {
                config = {
                    server_url: data.server_url || data.server_ip || "http://127.0.0.1:8000"
                };
                document.getElementById('serverUrl').value = config.server_url;
            })
            .catch(e => {
                console.log('Using default configuration');
            });
    }
}

// Save configuration
function saveConfig() {
    const serverUrl = document.getElementById('serverUrl').value.trim();
    if (!serverUrl) {
        showError('Please enter a valid server URL');
        return;
    }

    // Ensure URL has protocol
    let url = serverUrl;
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'http://' + url;
    }

    config.server_url = url;
    localStorage.setItem('outfitExtractorConfig', JSON.stringify(config));
    document.getElementById('serverUrl').value = config.server_url;
    showSuccess('Configuration saved successfully!');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    setupFileInput();
    setupDragAndDrop();
});

// Setup file input
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Setup drag and drop
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
}

// Handle file selection
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    selectedFile = file;
    const extractBtn = document.getElementById('extractBtn');
    extractBtn.disabled = false;

    // Preview original image
    const reader = new FileReader();
    reader.onload = (e) => {
        const originalImage = document.getElementById('originalImage');
        const originalPlaceholder = document.getElementById('originalPlaceholder');
        
        originalImage.src = e.target.result;
        originalImage.classList.add('show');
        originalPlaceholder.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Clear previous result
    const resultImage = document.getElementById('resultImage');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    resultImage.classList.remove('show');
    resultPlaceholder.style.display = 'block';
}

// Extract outfit
async function extractOutfit() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    const serverUrl = config.server_url;
    if (!serverUrl) {
        showError('Please configure server URL first');
        return;
    }

    // Show loading state
    showLoading(true);
    hideError();
    hideSuccess();

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${serverUrl}/extract`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        // Get image blob
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        // Display result
        const resultImage = document.getElementById('resultImage');
        const resultPlaceholder = document.getElementById('resultPlaceholder');
        
        resultImage.src = imageUrl;
        resultImage.classList.add('show');
        resultPlaceholder.style.display = 'none';

        showSuccess('Outfit extracted successfully!');
    } catch (error) {
        console.error('Extraction error:', error);
        showError(`Failed to extract outfit: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// UI Helper Functions
function showLoading(show) {
    const loading = document.getElementById('loading');
    const extractBtn = document.getElementById('extractBtn');
    
    if (show) {
        loading.classList.add('show');
        extractBtn.disabled = true;
    } else {
        loading.classList.remove('show');
        extractBtn.disabled = false;
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
    setTimeout(() => {
        errorDiv.classList.remove('show');
    }, 5000);
}

function hideError() {
    document.getElementById('error').classList.remove('show');
}

function showSuccess(message) {
    const successDiv = document.getElementById('success');
    successDiv.textContent = message;
    successDiv.classList.add('show');
    setTimeout(() => {
        successDiv.classList.remove('show');
    }, 3000);
}

function hideSuccess() {
    document.getElementById('success').classList.remove('show');
}
