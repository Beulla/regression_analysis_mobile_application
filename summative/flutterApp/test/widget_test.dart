import 'package:flutter_test/flutter_test.dart';
import 'package:exam_score_app/main.dart';

void main() {
  testWidgets('Predict screen loads', (WidgetTester tester) async {
    await tester.pumpWidget(const ExamScoreApp());
    expect(find.text('Predict'), findsOneWidget);
    expect(find.text('Hours Studied'), findsOneWidget);
  });
}
