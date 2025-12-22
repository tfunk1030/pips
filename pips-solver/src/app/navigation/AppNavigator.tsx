/**
 * App Navigation
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import HomeScreen from '../screens/HomeScreen';
import PuzzleViewerScreen from '../screens/PuzzleViewerScreen';
import SolveScreen from '../screens/SolveScreen';
import SettingsScreen from '../screens/SettingsScreen';
import OverlayBuilderScreen from '../screens/OverlayBuilderScreen';

export type RootStackParamList = {
  Home: undefined;
  Viewer: { puzzleId: string };
  Solve: { puzzleId: string };
  Settings: undefined;
  OverlayBuilder: { draftId?: string } | undefined;
};

const Stack = createStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <NavigationContainer>
        <Stack.Navigator
          screenOptions={{
            headerShown: false,
          }}
        >
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Viewer" component={PuzzleViewerScreen} />
          <Stack.Screen name="Solve" component={SolveScreen} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
          <Stack.Screen name="OverlayBuilder" component={OverlayBuilderScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}
