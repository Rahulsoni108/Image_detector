class AiImageDetector
  def predict(image_path)
    # Open and preprocess image: resize to (128,128), set RGB format
    image = MiniMagick::Image.open(image_path)
    image.resize "128x128!"
    image.format "jpg"

    # Retrieve and normalize pixel data
    image_data = image.get_pixels.flatten.map { |pixel| pixel / 255.0 }

    # Ensure correct array size for model input
    raise "Image data size mismatch" unless image_data.size == 128 * 128 * 3

    # Convert pixel data to JSON and pass to the prediction script
    json_data = image_data.to_json
    script_path = Rails.root.join('lib', 'scripts', 'predict_model.py').to_s
    command = "python3 #{script_path}"
    output, error, status = Open3.capture3(command, stdin_data: json_data)

    # Handle prediction response
    raise "Prediction failed: #{error}" unless status.success?

    output.strip.split.last
  end
end
