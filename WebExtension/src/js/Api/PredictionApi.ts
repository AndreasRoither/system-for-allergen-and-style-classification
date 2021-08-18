import axios, { AxiosResponse } from 'axios'
import { PredictRequest, PredictTypeRequest } from '~/types/PredictRequest'
import { PredictResponseItem } from '~/types/PredictResponseItem'
import { apiConfig } from '~/api/api.config'
import { PredictProbaResponse } from '~/types/PredictProbaResponse'

const axiosInstance = axios.create(apiConfig)

export default class PredictionApi {
  static predict_style(
    request: PredictRequest
  ): Promise<AxiosResponse<PredictResponseItem>> {
    return axiosInstance.request<PredictResponseItem>({
      method: 'post',
      url: '/predict_style',
      data: JSON.stringify(request),
    })
  }

  static predict_allergens(
    request: PredictRequest
  ): Promise<AxiosResponse<PredictProbaResponse>> {
    return axiosInstance.request<PredictProbaResponse>({
      method: 'post',
      url: '/predict_allergens',
      data: JSON.stringify(request),
    })
  }

  static predict_proba(
    request: PredictTypeRequest
  ): Promise<AxiosResponse<PredictProbaResponse>> {
    return axiosInstance.request<PredictProbaResponse>({
      method: 'post',
      url: '/predict_proba',
      data: JSON.stringify(request),
    })
  }
}
