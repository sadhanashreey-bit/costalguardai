# CoastGuard AI - Computer Vision

This module is responsible for detecting plastic waste and other waste using YOLO.

## Structure

datasets/
    raw/
    processed/

models/
    best.pt

outputs/

src/
    predict.py
    utils.py

classes.txt
README.md

## Detection Classes

plastic_waste
other_waste

## Pipeline

Dataset
    ↓
Dataset preparation
    ↓
YOLO training
    ↓
best.pt
    ↓
Image detection
    ↓
Video detection
    ↓
Live camera detection
    ↓
FastAPI integration

## Current Stage

Person 1 - Computer Vision

Current goal:
Train a waste detection model and create image inference.

## Model

The trained YOLO model should eventually be saved as:

