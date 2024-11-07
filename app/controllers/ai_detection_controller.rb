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

  def analyze
    if params[:image].blank?
      render json: { error: 'Image not provided' }, status: :bad_request
      return
    end

    response = SightengineService.new.check_image(params[:image]&.path)

    if response["status"] == "success"
      request_id = response["request"]["id"]
      ai_generated_score = response["type"]["ai_generated"]
      media_uri = response["media"]["uri"]

      result = ai_generated_score >= 0.5 ? 'AI-generated' : 'Real'
      # Do something with the data, e.g., save to the database, log it, etc.
      render json: {
        request_id: request_id,
        ai_generated_score: ai_generated_score,
        media_uri: media_uri,
        result: result
      }
    else
      render json: { error: 'Request failed' }, status: :unprocessable_entity
    end
  end
end
