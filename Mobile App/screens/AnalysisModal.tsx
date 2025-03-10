import React from 'react';
import { View, Text, StyleSheet, Button, ScrollView } from 'react-native';

interface Anomaly {
  timestamp: string;
  is_anomaly: string;
  score: number;
  metrics: {
    temperature: number;
    humidity: number;
    power: number;
    pressure: number;
  };
}

interface AnalysisData {
  anomalies: Anomaly[];
  total_energy_kwh: number;
  total_cost: number;
  average_daily_cost: number;
  peak_usage_kwh: number;
  peak_usage_cost: number;
  efficiency_score: number;
  llm_analysis: string[];
}

interface AnalysisModalProps {
  visible: boolean;
  onClose: () => void;
  analysisData: AnalysisData;
}

const AnalysisModal: React.FC<AnalysisModalProps> = ({ visible, onClose, analysisData }) => {
  if (!visible) return null;

  return (
    <View style={styles.modalContainer}>
      <View style={styles.modalContent}>
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <Text style={styles.title}>Analysis Results</Text>

          {/* Anomaly Detection Results */}
          <Text style={styles.sectionTitle}>Anomaly Detection</Text>
          {analysisData.anomalies && analysisData.anomalies.length > 0 ? (
            analysisData.anomalies.map((anomaly, index) => (
              <View key={index} style={styles.resultItem}>
                <Text>Timestamp: {anomaly.timestamp}</Text>
                <Text>Is Anomaly: {anomaly.is_anomaly}</Text>
                <Text>Score: {anomaly.score}</Text>
                <Text>Metrics: {JSON.stringify(anomaly.metrics)}</Text>
              </View>
            ))
          ) : (
            <Text>No anomalies detected.</Text>
          )}

          {/* Cost Analysis Results */}
          <Text style={styles.sectionTitle}>Cost Analysis</Text>
          <Text>Total Energy (kWh): {analysisData.total_energy_kwh}</Text>
          <Text>Total Cost: ${analysisData.total_cost}</Text>
          <Text>Average Daily Cost: ${analysisData.average_daily_cost}</Text>
          <Text>Peak Usage (kWh): {analysisData.peak_usage_kwh}</Text>
          <Text>Peak Usage Cost: ${analysisData.peak_usage_cost}</Text>
          <Text>Efficiency Score: {analysisData.efficiency_score}</Text>

          {/* LLM Analysis Results */}
          <Text style={styles.sectionTitle}>LLM Analysis</Text>
          {analysisData.llm_analysis && analysisData.llm_analysis.length > 0 ? (
            analysisData.llm_analysis.map((suggestion, index) => (
              <Text key={index}>{suggestion}</Text>
            ))
          ) : (
            <Text>No suggestions available.</Text>
          )}
        </ScrollView>
      </View>
      <Button title="Close" onPress={onClose} />
    </View>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 20,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    maxHeight: '80%', // Limit the height of the modal
  },
  scrollContainer: {
    flexGrow: 1, // Allow the ScrollView to grow
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
  },
  resultItem: {
    marginBottom: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
  },
});

export default AnalysisModal;