import { PredictProbaResponse } from '~/types/PredictProbaResponse'
import Swal, { SweetAlertIcon, SweetAlertResult } from 'sweetalert2'

class Queue<T> {
  private storage: T[] = [];

  constructor(private capacity: number = Infinity) {}

  enqueue(item: T): void {
    if (this.size() === this.capacity) {
      throw Error("Queue has reached max capacity, you cannot add more items");
    }
    this.storage.push(item);
  }
  dequeue(): T | undefined {
    return this.storage.shift();
  }
  size(): number {
    return this.storage.length;
  }
}

const GenericPopup = Swal.mixin({
  position: 'top-end',
  showConfirmButton: false,
  timer: 4000,
  timerProgressBar: true,
  didOpen: (toast) => {
    toast.addEventListener('mouseenter', Swal.stopTimer)
    toast.addEventListener('mouseleave', Swal.resumeTimer)
  },
})

export type PopupContent = { type: SweetAlertIcon | undefined, title: string | undefined, text: string }
const popups = new Queue<PopupContent>()
let popupOpen = false

/**
 * Add popup to the queue of existing ones; since only one popup can be shown at a time
 * @param content
 */
export function enqueuePopup(content: PopupContent) {
  console.log(content.text)

  popups.enqueue(content)

  if (popupOpen) return;
  popupOpen = true

  firePopup(content).then(() => {
    popupOpen = false;

    // get rid of current popup and get next one
    popups.dequeue()
    const next = popups.dequeue()

    if (next != undefined) {
      enqueuePopup(next)
    }
  })
}

/**
 * Show a popup with specified content
 * @param content
 */
function firePopup(content: PopupContent): Promise<SweetAlertResult<any>> {
  return GenericPopup.fire({
    position: 'top-end',
    icon: content.type,
    title: content.title,
    text: content.text,
  })
}

export const requestTitles = {
  selectionCopyRequested: 'selectionCopyRequested',
  clipboardCopyRequested: 'clipboardCopyRequested',
  siteCopyRequested: 'siteCopy',
  selectionCopied: 'selectionCopied',
  clipboardCopied: 'clipboardCopied',
  siteCopied: 'siteCopied',
} as const

export type CopyRequestMessage = { title: string }
export type CopiedMessage = { title: string, text: string }
export type Message = CopyRequestMessage | CopiedMessage

export function copiedClipboard(message: string): CopiedMessage {
  return {
    title: requestTitles.clipboardCopied,
    text: message,
  }
}

export function copiedSite(message: string): CopiedMessage {
  return {
    title: requestTitles.siteCopied,
    text: message,
  }
}

export function copiedSelection(message: string): CopiedMessage {
  return {
    title: requestTitles.selectionCopied,
    text: message,
  }
}

export function copySelectionRequest(): CopyRequestMessage {
  return {
    title: requestTitles.selectionCopyRequested,
  }
}

export function copySiteRequest(): CopyRequestMessage {
  return {
    title: requestTitles.siteCopyRequested,
  }
}

function hasTitle(value: { title?: unknown }): value is { title: string } {
  return typeof value.title == 'string'
}

export function isMessage(value: unknown): value is Message {
  return typeof value == 'object' && value != null && hasTitle(value)
}

/**
 * Convert to a message, since sending messages in the web api erases the type, so type casting is needed
 * @param value
 */
export function convertToMessage(value: unknown): Message {
  if (isMessage(value)) {
    return value
  } else {
    throw new Error(`${JSON.stringify(value)} is not a message`)
  }
}

/**
 * Send message to background script with information about selected content
 * and display popup depending on the returned content
 * @param message
 */
export function sendMessageToBackground(message: Message) {
  if (process.env.TARGET === 'chrome') {
    chrome.runtime.sendMessage(message, handleResponse)
  } else {
    browser.runtime.sendMessage(message)
      .then(value => {
        handleResponse(value)
      }).catch(reason => {
      handleError(reason)
    })
  }
}

/**
 * Send message to a specific tab to execute certain commands like copying and sending back the content
 * called from background script
 * @param message
 * @param tabId
 */
export function sendMessageToTab(message: Message, tabId: number) {
  if (process.env.TARGET === 'chrome') {
    chrome.tabs.sendMessage(tabId, message, handleResponse)
  } else {
    browser.tabs.sendMessage(tabId, message)
      .then(value => {
        handleResponse(value)
      }).catch(reason => {
      handleError(reason)
    })
  }
}

function roundToTwo(value: number) {
  return (Math.round(value * 100) / 100)
}

/**
 * Response is only shown in content script as the background script can not be used to show content
 * @param value
 */
function handleResponse(value: any) {

  if (!value) return
  let response = value as PredictProbaResponse

  if (response) {
    let text = response.predictions.map(p => `${p.name}: ${roundToTwo(+p.probability * 100)}%`).join(', ')

    if (response.customFilter.length > 0) {
      enqueuePopup({type: undefined, title: 'Custom filter triggered', text: response.customFilter.join(", ")})
    }

    if (text.length == 0) {
      text = 'No allergens found'
    }

    enqueuePopup({type: undefined, title: 'Allergen prediction', text: text})
  } else {
    handleError(value)
  }
}

function handleError(error: any) {
  console.log(`Error: ${error}`)

  if ('customFilter' in error && error.customFilter.length > 0) {
    enqueuePopup({type: undefined, title: 'Custom filter!', text: error.customFilter.join(", ")})
  }

  enqueuePopup({type: 'error', title: undefined, text: error.error})
}
