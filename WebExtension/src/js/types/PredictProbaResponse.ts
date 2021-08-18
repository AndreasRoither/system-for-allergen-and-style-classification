import { PredictResponseItem } from '~/types/PredictResponseItem'

export interface PredictProbaResponse {
  predictions: PredictResponseItem[],
  customFilter: string[]
}
