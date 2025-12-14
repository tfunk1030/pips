import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { LibraryScreen } from '../screens/LibraryScreen';
import { EditorScreen } from '../screens/EditorScreen';
import { PuzzleScreen } from '../screens/PuzzleScreen';
import { SettingsScreen } from '../screens/SettingsScreen';

export type RootStackParamList = {
  Library: undefined;
  Editor: { initialText?: string } | undefined;
  Puzzle: { puzzleId?: string; puzzleText?: string };
  Settings: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export function RootNavigator() {
  return (
    <Stack.Navigator initialRouteName="Library">
      <Stack.Screen name="Library" component={LibraryScreen} options={{ title: 'Pips CSP' }} />
      <Stack.Screen name="Editor" component={EditorScreen} options={{ title: 'Paste YAML/JSON' }} />
      <Stack.Screen name="Puzzle" component={PuzzleScreen} options={{ title: 'Puzzle' }} />
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: 'Settings' }} />
    </Stack.Navigator>
  );
}



