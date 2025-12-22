import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import PuzzleEditorScreen from '../screens/PuzzleEditorScreen';
import PuzzleViewerScreen from '../screens/PuzzleViewerScreen';
import SolveScreen from '../screens/SolveScreen';
import SettingsScreen from '../screens/SettingsScreen';
import ExtractionResultScreen from '../screens/ExtractionResultScreen';
import { PuzzleSpec } from '../../model/types';
import { ExtractionResult } from '../../extraction/types';

export type RootStackParamList = {
  Home: undefined;
  Editor: { initialText?: string } | undefined;
  Viewer: { puzzle: PuzzleSpec; sourceText: string };
  Solve: { puzzle: PuzzleSpec; sourceText: string };
  Settings: undefined;
  ExtractionResult: { extractionResult: ExtractionResult; sourceImageUri?: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const RootNavigator = () => (
  <Stack.Navigator
    initialRouteName="Home"
    screenOptions={{ headerStyle: { backgroundColor: '#11162a' }, headerTintColor: '#e6e6e6' }}
  >
    <Stack.Screen name="Home" component={HomeScreen} options={{ title: 'Pips Library' }} />
    <Stack.Screen name="Editor" component={PuzzleEditorScreen} options={{ title: 'Import Puzzle' }} />
    <Stack.Screen name="Viewer" component={PuzzleViewerScreen} options={{ title: 'Puzzle' }} />
    <Stack.Screen name="Solve" component={SolveScreen} options={{ title: 'Solve' }} />
    <Stack.Screen name="Settings" component={SettingsScreen} />
    <Stack.Screen name="ExtractionResult" component={ExtractionResultScreen} options={{ title: 'Extraction Results' }} />
  </Stack.Navigator>
);

export default RootNavigator;
