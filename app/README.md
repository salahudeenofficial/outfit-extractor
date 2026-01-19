# Outfit Extractor - Web Client

This is the web client application for the Outfit Extractor API. It provides a user-friendly interface to upload images and extract outfits.

## Features

- **Server Configuration**: Easily configure the server IP address and port
- **Image Upload**: Drag and drop or click to upload images
- **Real-time Preview**: View original and extracted outfit images side by side
- **Responsive Design**: Works on desktop and mobile devices

## Usage

1. Open `index.html` in a web browser
2. Configure the server URL (default: `http://127.0.0.1:8000`)
3. Click "Save Configuration"
4. Upload an image by clicking the upload area or dragging and dropping
5. Click "Extract Outfit" to process the image
6. View the extracted outfit in the result preview

## Configuration

The server URL can be configured in two ways:

1. **Via Web Interface**: Enter the server URL in the configuration section and click "Save"
2. **Via config.json**: Edit `config.json` and set the `server_url` field

The configuration is saved in browser localStorage for persistence.

## Server Requirements

The client expects the API server to be running and accessible at the configured URL. The server should expose:

- `POST /extract` - Endpoint for outfit extraction
- `GET /health` - Health check endpoint

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## Troubleshooting

- **Connection Error**: Ensure the server is running and the URL is correct
- **CORS Error**: Make sure the server has CORS enabled for your domain
- **Image Not Loading**: Check that the server is returning a valid PNG image
