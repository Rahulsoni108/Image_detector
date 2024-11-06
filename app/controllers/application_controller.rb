# frozen_string_literal: true

class ApplicationController < ActionController::Base
  protect_from_forgery
  rescue_from ActiveRecord::RecordNotFound, with: :not_found

  private

  def not_found
    render json: { 'errors' => ['Record not found'] }, status: :not_found
  end
end
