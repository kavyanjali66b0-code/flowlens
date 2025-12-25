"""
Enhanced Semantic Analysis Example

This example demonstrates how to use the enhanced semantic analyzer
with ML/NLP capabilities for comprehensive code analysis.
"""

import logging
from enhanced_semantic_analyzer import EnhancedSemanticAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_react_component():
    """Example: Analyze a React component with enhanced semantic analysis."""
    
    # Sample React component code
    code = """
    import React, { useState, useEffect, useContext } from 'react';
    import { UserContext } from './UserContext';
    import { validateEmail } from './utils/validation';
    
    function UserProfile({ userId, onUpdate }) {
        const { user, setUser } = useContext(UserContext);
        const [isLoading, setIsLoading] = useState(false);
        const [error, setError] = useState(null);
        
        useEffect(() => {
            const fetchUserData = async () => {
                setIsLoading(true);
                try {
                    const response = await fetch(`/api/users/${userId}`);
                    if (!response.ok) {
                        throw new Error('Failed to fetch user');
                    }
                    const userData = await response.json();
                    setUser(userData);
                } catch (err) {
                    setError(err.message);
                } finally {
                    setIsLoading(false);
                }
            };
            
            if (userId) {
                fetchUserData();
            }
        }, [userId, setUser]);
        
        const handleEmailUpdate = (newEmail) => {
            if (!validateEmail(newEmail)) {
                setError('Invalid email format');
                return;
            }
            
            onUpdate({ email: newEmail });
        };
        
        if (isLoading) return <div>Loading...</div>;
        if (error) return <div>Error: {error}</div>;
        if (!user) return <div>No user found</div>;
        
        return (
            <div className="user-profile">
                <h1>{user.name}</h1>
                <p>Email: {user.email}</p>
                <button onClick={() => handleEmailUpdate('new@example.com')}>
                    Update Email
                </button>
            </div>
        );
    }
    
    export default UserProfile;
    """
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSemanticAnalyzer(
        enable_embeddings=True,
        enable_intent_classification=True,
        enable_pattern_detection=True,
        enable_behavior_prediction=True,
        cache_results=True
    )
    
    # Perform analysis
    logger.info("Starting enhanced semantic analysis...")
    result = analyzer.analyze(code, "components/UserProfile.jsx", "javascript")
    
    # Display results
    logger.info(f"Analysis completed in {result.analysis_time_ms:.2f}ms")
    
    # Intent Classifications
    logger.info("\n=== Intent Classifications ===")
    for intent in result.intent_classifications:
        logger.info(f"Function: {intent.function_name}")
        logger.info(f"  Primary Intent: {intent.primary_intent.value}")
        logger.info(f"  Confidence: {intent.confidence:.2f}")
        logger.info(f"  Evidence: {', '.join(intent.evidence[:2])}")
        logger.info("")
    
    # Detected Patterns
    logger.info("=== Detected Patterns ===")
    for pattern in result.detected_patterns:
        logger.info(f"Pattern: {pattern.pattern_type.value}")
        logger.info(f"  Confidence: {pattern.confidence:.2f}")
        logger.info(f"  Components: {pattern.components}")
        logger.info(f"  Evidence: {', '.join(pattern.evidence[:2])}")
        logger.info("")
    
    # Behavior Predictions
    logger.info("=== Behavior Predictions ===")
    for behavior in result.behavior_predictions:
        logger.info(f"Entity: {behavior.entity_name}")
        logger.info(f"  Predicted Behaviors:")
        for behavior_type, confidence in behavior.predicted_behaviors.items():
            if confidence > 0.5:
                logger.info(f"    {behavior_type.value}: {confidence:.2f}")
        if behavior.risks:
            logger.info(f"  Risks: {', '.join(behavior.risks[:2])}")
        logger.info("")
    
    # ML Insights
    logger.info("=== ML Insights ===")
    insights = result.ml_insights
    logger.info(f"Semantic Clusters: {insights.get('semantic_clusters_count', 0)}")
    logger.info(f"Intent Distribution: {insights.get('intent_distribution', {})}")
    logger.info(f"Pattern Summary: {insights.get('pattern_summary', {})}")
    logger.info(f"Behavior Risks: {insights.get('behavior_risks', [])}")
    
    # Recommendations
    recommendations = insights.get('recommendations', [])
    if recommendations:
        logger.info("\n=== Recommendations ===")
        for i, rec in enumerate(recommendations[:5], 1):
            logger.info(f"{i}. {rec}")
    
    return result


def analyze_backend_service():
    """Example: Analyze a backend service with enhanced semantic analysis."""
    
    # Sample backend service code
    code = """
    const express = require('express');
    const { body, validationResult } = require('express-validator');
    const UserRepository = require('./repositories/UserRepository');
    const EmailService = require('./services/EmailService');
    const logger = require('./utils/logger');
    
    class UserController {
        constructor(userRepository, emailService) {
            this.userRepository = userRepository;
            this.emailService = emailService;
        }
        
        async createUser(req, res) {
            try {
                // Validate input
                const errors = validationResult(req);
                if (!errors.isEmpty()) {
                    return res.status(400).json({ errors: errors.array() });
                }
                
                const { name, email, password } = req.body;
                
                // Check if user already exists
                const existingUser = await this.userRepository.findByEmail(email);
                if (existingUser) {
                    return res.status(409).json({ error: 'User already exists' });
                }
                
                // Hash password
                const hashedPassword = await bcrypt.hash(password, 10);
                
                // Create user
                const user = await this.userRepository.create({
                    name,
                    email,
                    password: hashedPassword
                });
                
                // Send welcome email
                await this.emailService.sendWelcomeEmail(user.email, user.name);
                
                logger.info(`User created: ${user.id}`);
                
                res.status(201).json({
                    id: user.id,
                    name: user.name,
                    email: user.email
                });
                
            } catch (error) {
                logger.error('Error creating user:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        }
        
        async getUser(req, res) {
            try {
                const { id } = req.params;
                const user = await this.userRepository.findById(id);
                
                if (!user) {
                    return res.status(404).json({ error: 'User not found' });
                }
                
                res.json({
                    id: user.id,
                    name: user.name,
                    email: user.email
                });
                
            } catch (error) {
                logger.error('Error fetching user:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        }
    }
    
    module.exports = UserController;
    """
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSemanticAnalyzer(
        enable_embeddings=True,
        enable_intent_classification=True,
        enable_pattern_detection=True,
        enable_behavior_prediction=True
    )
    
    # Perform analysis
    logger.info("Starting backend service analysis...")
    result = analyzer.analyze(code, "controllers/UserController.js", "javascript")
    
    # Display results
    logger.info(f"Analysis completed in {result.analysis_time_ms:.2f}ms")
    
    # Show confidence scores
    logger.info("\n=== Confidence Scores ===")
    for aspect, score in result.confidence_scores.items():
        logger.info(f"{aspect}: {score:.2f}")
    
    # Show enhanced relationships
    logger.info(f"\n=== Enhanced Relationships ===")
    logger.info(f"Semantic Similarity Edges: {len(result.semantic_similarity_edges)}")
    logger.info(f"Intent-based Edges: {len(result.intent_based_edges)}")
    logger.info(f"Pattern-based Edges: {len(result.pattern_based_edges)}")
    
    return result


def main():
    """Main function to run examples."""
    logger.info("Enhanced Semantic Analysis Examples")
    logger.info("=" * 50)
    
    # Analyze React component
    logger.info("\n1. Analyzing React Component...")
    react_result = analyze_react_component()
    
    # Analyze backend service
    logger.info("\n2. Analyzing Backend Service...")
    backend_result = analyze_backend_service()
    
    # Show analyzer statistics
    logger.info("\n=== Analyzer Statistics ===")
    stats = react_result.ml_insights.get('analyzer_stats', {})
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()

