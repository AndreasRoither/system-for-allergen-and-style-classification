
<h1 align="center">
  <!--<a name="logo" href=""><img src="" alt="Logo" width="200"></a>-->
  <br>
  Web Extension Documentation
</h1>

## Overview

The main goal of the browser extension is to use the trained classifiers from the REST API to predict allergen and style in text by showing the resulting information to the user. Additionally, a custom filter list can be used to get extra notifications.
The browser extension uses [Parcel](https://parceljs.org) and [React](https://reactjs.org) with [TypeScript](https://www.typescriptlang.org). The main development is focused on the Firefox browser, but additional options have been added that allows development for the Chrome browser as well.

## Startup

Before starting, the build for the respective platform has to be executed at least once by using the command:

```shell
yarn build:firefox|chrome
```

Then the extension can be started using:  

```shell
yarn firefox|chrome
```

## Libraries

### [Axios](https://github.com/axios/axios)

Promise based HTTP client for the browser and node.js.  

### [parcel-plugin-web-extension](https://github.com/parcel-bundler/parcel

This parcel plugin enables you to use a WebExtension manifest.json as an entry point. For more information about manifest.json, please refer to the [MDN](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json) docs.

### [parcel-plugin-compress](https://github.com/ralscha/parcel-plugin-compress)

Parcel plugin that pre-compresses all assets in production mode.

This plugin utilizes [@gfx/zopfli](https://github.com/gfx/universal-zopfli-js), [node-zopfli-es](https://github.com/jaeh/node-zopfli-es) and zlib for GZip compression and zlib (Node 11.7.0+) and [brotli](https://www.npmjs.com/package/brotli) for Brotli compression.

### [crx3](https://github.com/ahwayakchih/crx3)

This module allows you to create web extension files for Chromium and Google Chrome browsers.
It creates CRXv3 files, which work for Chrome versions 64.0.3242 and above.
If you need to create web extension file for older browser versions, where CRXv3 is not supported, use CRX module instead.
It requires Node.js version 12 (or above) and runs on Linux, MacOS and Windows.

### [styled-components](https://github.com/styled-components/styled-components)

Visual primitives for the component age. Use the best bits of ES6 and CSS to style your apps without stress.
