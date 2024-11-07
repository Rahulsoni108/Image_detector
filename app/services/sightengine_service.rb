require 'httparty'

class SightengineService
  include HTTParty
  base_uri 'https://api.sightengine.com/1.0'

  def initialize
    @api_user = ENV['SIGHTENGINE_API_USER']
    @api_secret = ENV['SIGHTENGINE_API_SECRET']
  end

  def check_image(image_path)
    options = {
      body: {
        models: 'genai',
        api_user: @api_user,
        api_secret: @api_secret,
        media: File.open(image_path) # Attach the file here with the correct key
      }
    }

    # Use multipart: true to indicate that we are sending a multipart request
    response = self.class.post('/check.json', options)
    JSON.parse(response.body)
  end
end
