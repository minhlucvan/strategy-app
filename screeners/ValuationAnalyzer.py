import requests
import pandas as pd
import numpy as np
import streamlit as st
from utils.sector_analysis import SectorMetrics, SectorType
from enum import Enum

class ValuationAnalyzer:
    def __init__(self):
        self.sector_metrics = SectorMetrics()
        
        # Base weights (60% of total score)
        self.base_weights = {
            'P/E Score': 0.15,      
            'P/B Score': 0.10,      
            'PEG Score': 0.15,      
            'DCF Score': 0.20       
        }
        
        # Technical weights (20% of total score)
        self.technical_weights = {
            'RSI_Score': 0.05,     
            'MA_Score': 0.05,      
            'Growth_Score': 0.10    
        }

    def analyze_valuation(self, metrics, sector):
        """
        Analyze stock valuation using multiple metrics and return a comprehensive score
        Score range: 0 (extremely overvalued) to 100 (extremely undervalued)
        """
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        scores = {}
        
        # Get industry averages for the sector
        industry_pe = self.sector_metrics.INDUSTRY_PE[sector_type]
        industry_pb = self.sector_metrics.INDUSTRY_PB[sector_type]
        
        # Calculate scores using industry-specific benchmarks
        if metrics['P/E Ratio'] and metrics['P/E Ratio'] > 0:
            scores['P/E Score'] = self._score_pe_ratio(metrics['P/E Ratio'], industry_pe)
            
        if metrics['P/B Ratio'] and metrics['P/B Ratio'] > 0:
            scores['P/B Score'] = self._score_pb_ratio(metrics['P/B Ratio'], industry_pb)
        
        # 3. PEG Ratio Analysis (Weight: 20%)
        if metrics['PEG Ratio'] and metrics['PEG Ratio'] > 0:
            peg_score = self._score_peg_ratio(metrics['PEG Ratio'])
            scores['PEG Score'] = peg_score
        
        # 4. DCF Valuation Analysis (Weight: 30%)
        if metrics['Current Price'] and metrics['DCF Value']:
            dcf_score = self._score_dcf_value(metrics['Current Price'], metrics['DCF Value'])
            scores['DCF Score'] = dcf_score
        
        # 5. Financial Health Score (Weight: 10%)
        health_score = self._score_financial_health(metrics)
        scores['Financial Health Score'] = health_score
        
        # Calculate weighted average score
        weights = {
            'P/E Score': 0.25,
            'P/B Score': 0.15,
            'PEG Score': 0.20,
            'DCF Score': 0.30,
            'Financial Health Score': 0.10
        }
        
        final_score = 0
        valid_weights_sum = 0
        
        for metric, score in scores.items():
            if score is not None:
                final_score += score * weights[metric]
                valid_weights_sum += weights[metric]
        
        if valid_weights_sum > 0:
            final_score = final_score / valid_weights_sum
        
        # Generate recommendation
        recommendation = self.get_recommendation(metrics, final_score)  # Changed from _get_recommendation
            
        return {
            'detailed_scores': scores,
            'final_score': final_score,
            'valuation_status': self._get_valuation_status(final_score),
            'recommendation': recommendation
        }

    def _calculate_technical_score(self, metrics):
        """Calculate technical analysis score"""
        score = 50  # Start at neutral
    
        # RSI Analysis
        rsi = metrics.get('RSI')
        if rsi is not None:
            if rsi < 30:  # Oversold
                score += 20
            elif rsi < 40:
                score += 10
            elif rsi > 70:  # Overbought
                score -= 20
            elif rsi > 60:
                score -= 10
    
        # Moving Average Analysis
        ma50 = metrics.get('MA50')
        ma200 = metrics.get('MA200')
        if ma50 is not None and ma200 is not None:
            if ma50 > ma200:  # Golden Cross
                score += 15
            else:  # Death Cross
                score -= 15
    
        # Volume Analysis
        vol_avg = metrics.get('Volume_Average')
        vol_current = metrics.get('Volume_Current')
        if vol_avg is not None and vol_current is not None:
            if vol_current > vol_avg * 1.5:  # High volume
                score += 10
    
        return max(0, min(100, score))

    def _calculate_sector_score(self, metrics, sector):
        """Calculate sector-specific score"""
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        score = 50  # Start at neutral
        sector_metrics = self.sector_metrics.BASE_METRICS.copy()
        
        # Add sector-specific metrics if available
        if sector_type in self.sector_metrics.SECTOR_CONFIGS:
            sector_metrics.update(
                self.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
            )
        
        total_weight = 0
        weighted_score = 0
        
        for metric_name, config in sector_metrics.items():
            value = metrics.get(metric_name)
            if value is not None and config['threshold'] is not None:
                # Calculate metric score
                metric_score = self._score_metric(value, config['threshold'])
                weighted_score += metric_score * config['weight']
                total_weight += config['weight']
        
        # Adjust for risk factors
        risk_adjustment = self._calculate_risk_adjustment(metrics, sector_type)
        
        # Calculate final score
        if total_weight > 0:
            final_score = (weighted_score / total_weight) * 0.8 + risk_adjustment * 0.2
            return max(0, min(100, final_score))
        
        return score

    def _score_metric(self, value, threshold):
        """Convert metric value to score between 0 and 100"""
        if threshold == 0:
            return 50
        
        relative_performance = value / threshold
        if relative_performance >= 2:
            return 100
        elif relative_performance >= 1:
            return 75 + (relative_performance - 1) * 25
        else:
            return max(0, relative_performance * 75)

    def _calculate_risk_adjustment(self, metrics, sector_type):
        """Calculate risk adjustment based on sector-specific risk factors"""
        risk_score = 50  # Neutral starting point
        
        # Get sector-specific risk factors
        sector_config = self.sector_metrics.SECTOR_CONFIGS.get(sector_type)
        if not sector_config or 'risk_factors' not in sector_config:
            return risk_score
        
        risk_factors = sector_config['risk_factors']
        
        # Technology sector risk adjustments
        if sector_type == SectorType.TECHNOLOGY:
            # Tech obsolescence risk
            if metrics.get('R&D_Ratio', 0) < 0.05:
                risk_score -= 10
            elif metrics.get('R&D_Ratio', 0) > 0.15:
                risk_score += 10
                
            # Cybersecurity risk
            if metrics.get('Security_Investment_Ratio', 0) > 0.05:
                risk_score += 5
                
            # Competition risk
            if metrics.get('Market_Share', 0) > 0.20:
                risk_score += 10
                
        # Financial sector risk adjustments
        elif sector_type == SectorType.FINANCIAL:
            # Interest rate risk
            if metrics.get('Interest_Rate_Sensitivity', 0) > 0.5:
                risk_score -= 10
                
            # Credit risk
            if metrics.get('NPL_Ratio', 0) < 0.02:
                risk_score += 10
            elif metrics.get('NPL_Ratio', 0) > 0.05:
                risk_score -= 10
                
            # Capital adequacy
            if metrics.get('Capital_Adequacy_Ratio', 0) > 0.12:
                risk_score += 10
                
        # Healthcare sector risk adjustments
        elif sector_type == SectorType.HEALTHCARE:
            # Regulatory risk
            if metrics.get('Regulatory_Compliance_Score', 0) > 0.8:
                risk_score += 10
                
            # Patent expiry risk
            if metrics.get('Patent_Protection_Years', 0) > 5:
                risk_score += 5
                
            # Clinical trial risk
            if metrics.get('Clinical_Trial_Success_Rate', 0) > 0.6:
                risk_score += 10
                
        # Energy sector risk adjustments
        elif sector_type == SectorType.ENERGY:
            # Environmental risk
            if metrics.get('ESG_Score', 0) > 70:
                risk_score += 10
                
            # Resource depletion risk
            if metrics.get('Reserve_Life', 0) > 15:
                risk_score += 10
                
            # Production efficiency
            if metrics.get('Production_Cost', 0) < self.sector_metrics.BASE_METRICS['Operating_Margin']['threshold']:
                risk_score += 5
        
        # Add similar conditions for other sectors...
        
        return max(0, min(100, risk_score))

    def get_sector_metrics_breakdown(self, metrics, sector_type):
        """Get detailed breakdown of sector-specific metrics"""
        breakdown = {
            'base_metrics': {},
            'sector_specific': {},
            'risk_factors': {}
        }
        
        # Add base metrics
        for metric, config in self.sector_metrics.BASE_METRICS.items():
            value = metrics.get(metric)
            if value is not None:
                breakdown['base_metrics'][metric] = {
                    'value': value,
                    'threshold': config['threshold'],
                    'score': self._score_metric(value, config['threshold'])
                }
        
        # Add sector-specific metrics
        sector_config = self.sector_metrics.SECTOR_CONFIGS.get(sector_type)
        if sector_config and 'specific_metrics' in sector_config:
            for metric, config in sector_config['specific_metrics'].items():
                value = metrics.get(metric)
                if value is not None:
                    breakdown['sector_specific'][metric] = {
                        'value': value,
                        'threshold': config['threshold'],
                        'score': self._score_metric(value, config['threshold'])
                    }
        
        # Add risk factors
        if sector_config and 'risk_factors' in sector_config:
            for risk_factor in sector_config['risk_factors']:
                value = metrics.get(f'{risk_factor}_Risk')
                if value is not None:
                    breakdown['risk_factors'][risk_factor] = value
        
        return breakdown

    def _score_pe_ratio(self, pe_ratio, industry_avg):
        """Score P/E ratio relative to industry average"""
        if pe_ratio <= 0:
            return None
            
        if pe_ratio < industry_avg:
            return min(100, (1 - (pe_ratio / industry_avg)) * 100 + 50)
        else:
            return max(0, (2 - (pe_ratio / industry_avg)) * 50)
    
    def _score_pb_ratio(self, pb_ratio, industry_avg):
        """Score P/B ratio relative to industry average"""
        if pb_ratio <= 0:
            return None
            
        if pb_ratio < industry_avg:
            return min(100, (1 - (pb_ratio / industry_avg)) * 100 + 50)
        else:
            return max(0, (2 - (pb_ratio / industry_avg)) * 50)
    
    def _score_peg_ratio(self, peg_ratio):
        """Score PEG ratio (1 is considered fair value)"""
        if peg_ratio <= 0:
            return None
            
        if peg_ratio < 1:
            return min(100, (1 - peg_ratio) * 50 + 50)
        else:
            return max(0, (2 - peg_ratio) * 50)
    
    def _score_dcf_value(self, current_price, dcf_value):
        """Score based on DCF valuation"""
        if current_price <= 0 or dcf_value <= 0:
            return None
            
        ratio = current_price / dcf_value
        if ratio < 1:
            return min(100, (1 - ratio) * 100 + 50)
        else:
            return max(0, (2 - ratio) * 50)
    
    def _score_financial_health(self, metrics):
        """Score overall financial health"""
        score = 50  # Start at neutral
        
        # Check Free Cash Flow
        if metrics.get('FCF'):
            if metrics['FCF'] > 0:
                score += 10
            else:
                score -= 10
        
        # Check Debt/Equity
        if metrics.get('Debt/Equity'):
            if metrics['Debt/Equity'] < 1:
                score += 10
            elif metrics['Debt/Equity'] > 2:
                score -= 10
        
        return max(0, min(100, score))
    
    def _get_valuation_status(self, score):
        """Convert numerical score to valuation status"""
        if score >= 80:
            return "Significantly Undervalued"
        elif score >= 60:
            return "Moderately Undervalued"
        elif score >= 40:
            return "Fairly Valued"
        elif score >= 20:
            return "Moderately Overvalued"
        else:
            return "Significantly Overvalued"
    
    def get_recommendation(self, metrics, score):  # Changed from _get_recommendation to get_recommendation
        """Generate investment recommendation based on multiple factors"""
        # Base recommendation on valuation score
        if score >= 80:
            base_rec = "Strong Buy"
        elif score >= 60:
            base_rec = "Buy"
        elif score >= 40:
            base_rec = "Hold"
        elif score >= 20:
            base_rec = "Sell"
        else:
            base_rec = "Strong Sell"
        
        # Adjust recommendation based on additional factors
        adjustment_points = 0
        
        # Financial Health Adjustments
        fcf = metrics.get('FCF')
        if fcf is not None and fcf > 0:
            adjustment_points += 1

        debt_equity = metrics.get('Debt/Equity')
        if debt_equity is not None and debt_equity < 1:
            adjustment_points += 1
        
        # Risk Adjustments
        beta = metrics.get('Beta')
        if beta is not None:
            if beta < 0.8:  # Low volatility
                adjustment_points += 1
            elif beta > 1.5:  # High volatility
                adjustment_points -= 1
        
        # Dividend Consideration
        div_yield = metrics.get('Dividend Yield')
        if div_yield is not None and div_yield > 0.02:  # 2% yield threshold
            adjustment_points += 1
        
        # DCF Value vs Current Price
        current_price = metrics.get('Current Price')
        dcf_value = metrics.get('DCF Value')
        if current_price is not None and dcf_value is not None and current_price > 0:
            dcf_premium = (dcf_value - current_price) / current_price
            if dcf_premium > 0.3:  # 30% upside
                adjustment_points += 1
            elif dcf_premium < -0.3:  # 30% downside
                adjustment_points -= 1
        
        # Adjust final recommendation based on points
        rec_scale = {
            "Strong Buy": 2,
            "Buy": 1,
            "Hold": 0,
            "Sell": -1,
            "Strong Sell": -2
        }
        
        base_value = rec_scale[base_rec]
        adjusted_value = base_value + (adjustment_points * 0.5)  # Scale adjustment impact
        
        # Convert back to recommendation
        for rec, value in rec_scale.items():
            if adjusted_value >= value - 0.25:
                return rec
        
        return base_rec
