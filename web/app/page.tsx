"use client";

import { useState } from "react";
import { TextAnalyzer } from "../components/TextAnalyzer";
import { Header } from "../components/Header";
import { Footer } from "../components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Sloposcope
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Detect AI-generated text patterns and measure "slop" across
              multiple dimensions
            </p>
          </div>

          <TextAnalyzer />
        </div>
      </main>
      <Footer />
    </div>
  );
}
