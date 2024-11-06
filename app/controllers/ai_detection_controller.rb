class AiDetectionController < ApplicationController
  def detect
    uploaded_image = params[:image]
    if uploaded_image.nil?
      render json: { error: 'No image provided' }, status: :unprocessable_entity
      return
    end

    # Preprocess the image
    image_path = uploaded_image.tempfile.path
    # processed_image = ImagePreprocessor.preprocess(image_path)

    # Predict using the detector
    detector = AiImageDetector.new
    result = detector.predict(image_path)

    render json: { result: result }
  end
end
