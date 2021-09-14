
<h1 align="center">
  <!--<a name="logo" href=""><img src="" alt="Logo" width="200"></a>-->
  <br>
  System for allergen and style classification

  ![Contributions welcome](https://img.shields.io/badge/contributions-welcome-green.svg)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
</h1>

## Overview

This project is structured in 3 parts each in their own folder. Each has their own `README.md` file:

- Machine learning models and analysis
- Model REST Api
- Web Extension

## Demo

![Demo Gif](./images/demo.gif)

## Project Description

This project has been created as part of a master thesis. The proposed system should include/requires the following:

- Style classification of recipes
- Allergen classification of recipes
- Warnings for custom ingredients
- Architecture that allows extensions regardless of platform
- Browser extension for a popular browser like Firefox

## Motivation

The number of available food products on the market and the lack of food databases with correct allergen labelling make it challenging to provide accurate food allergen classification.
In addition, the sheer amount of different ingredients that can contain food allergens, can be hard to memorize. Although affected individuals are likely to know about ingredients that affect them, some might slip through their attention when reading and cause unnecessary stress.
Individuals that have allergic reactions to certain food allergens have to be careful when looking for recipes online, for some it is even a matter of life and death.
Given the rise in the prevalence of food allergy [@tang2017food], additional measures to inform individuals with food allergies become more and more important. A system that aids in the selection of recipes could reduce the amount of time as well as the frustration of affected individuals. Additionally, the integration with existing technologies like a web browser, will reduce time and help with scanning instead of having a dedicated app where text has to be copied in.

## Technology used

IDE:  
[IntelliJ IDEA](https://www.jetbrains.com/idea/)  

Other tools/libraries:  
[React](https://reactjs.org/)  
[Parcel](https://parceljs.org/)  
[Python](https://www.python.org/)  
[Docker](https://www.docker.com/)  
[scikit-learn](https://scikit-learn.org/)
