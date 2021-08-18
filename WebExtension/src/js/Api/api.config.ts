import { PredictRequest, PredictTypeRequest } from '~/types/PredictRequest'

export const apiConfig = {
  returnRejectedPromiseOnError: true,
  timeout: 5000,
  baseURL: process.env.HOST,
  headers: {
    'Content-type': 'application/json',
  },
}