import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

const String apiBase =
    'https://regression-analysis-mobile-application-vs8p.onrender.com';

void main() {
  runApp(const ExamScoreApp());
}

class ExamScoreApp extends StatelessWidget {
  const ExamScoreApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Exam Score Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        ),
      ),
      home: const PredictScreen(),
    );
  }
}

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  final _hoursCtrl = TextEditingController();
  final _attendanceCtrl = TextEditingController();
  final _prevScoresCtrl = TextEditingController();
  final _tutoringCtrl = TextEditingController();
  final _parentalInvCtrl = TextEditingController();
  final _resourcesCtrl = TextEditingController();
  final _motivationCtrl = TextEditingController();
  final _incomeCtrl = TextEditingController();
  final _teacherCtrl = TextEditingController();
  final _peerCtrl = TextEditingController();
  final _parentEduCtrl = TextEditingController();
  final _distanceCtrl = TextEditingController();
  final _extraCtrl = TextEditingController();
  final _internetCtrl = TextEditingController();
  final _disabilityCtrl = TextEditingController();

  String _resultText = '';
  bool _isError = false;
  bool _loading = false;

  @override
  void dispose() {
    _hoursCtrl.dispose();
    _attendanceCtrl.dispose();
    _prevScoresCtrl.dispose();
    _tutoringCtrl.dispose();
    _parentalInvCtrl.dispose();
    _resourcesCtrl.dispose();
    _motivationCtrl.dispose();
    _incomeCtrl.dispose();
    _teacherCtrl.dispose();
    _peerCtrl.dispose();
    _parentEduCtrl.dispose();
    _distanceCtrl.dispose();
    _extraCtrl.dispose();
    _internetCtrl.dispose();
    _disabilityCtrl.dispose();
    super.dispose();
  }

  String? _checkMissing() {
    final fields = {
      'Hours Studied': _hoursCtrl.text,
      'Attendance': _attendanceCtrl.text,
      'Previous Scores': _prevScoresCtrl.text,
      'Tutoring Sessions': _tutoringCtrl.text,
      'Parental Involvement': _parentalInvCtrl.text,
      'Access to Resources': _resourcesCtrl.text,
      'Motivation Level': _motivationCtrl.text,
      'Family Income': _incomeCtrl.text,
      'Teacher Quality': _teacherCtrl.text,
      'Peer Influence': _peerCtrl.text,
      'Parental Education Level': _parentEduCtrl.text,
      'Distance from Home': _distanceCtrl.text,
      'Extracurricular Activities': _extraCtrl.text,
      'Internet Access': _internetCtrl.text,
      'Learning Disabilities': _disabilityCtrl.text,
    };

    final empty = <String>[];
    fields.forEach((name, value) {
      if (value.trim().isEmpty) empty.add(name);
    });

    if (empty.isEmpty) return null;
    return 'Please fill in: ${empty.join(', ')}';
  }

  String? _checkRanges() {
    final hours = double.tryParse(_hoursCtrl.text.trim());
    final attendance = double.tryParse(_attendanceCtrl.text.trim());
    final prev = double.tryParse(_prevScoresCtrl.text.trim());
    final tutoring = int.tryParse(_tutoringCtrl.text.trim());

    if (hours == null) return 'Hours Studied must be a number';
    if (hours < 0 || hours > 60) {
      return 'Hours Studied is out of range (0 - 60)';
    }

    if (attendance == null) return 'Attendance must be a number';
    if (attendance < 0 || attendance > 100) {
      return 'Attendance is out of range (0 - 100)';
    }

    if (prev == null) return 'Previous Scores must be a number';
    if (prev < 0 || prev > 100) {
      return 'Previous Scores is out of range (0 - 100)';
    }

    if (tutoring == null) return 'Tutoring Sessions must be a whole number';
    if (tutoring < 0 || tutoring > 20) {
      return 'Tutoring Sessions is out of range (0 - 20)';
    }

    const lowMedHigh = ['Low', 'Medium', 'High'];
    const peerOpts = ['Negative', 'Neutral', 'Positive'];
    const eduOpts = ['High School', 'College', 'Postgraduate'];
    const distOpts = ['Near', 'Moderate', 'Far'];
    const yesNo = ['Yes', 'No'];

    if (!lowMedHigh.contains(_parentalInvCtrl.text.trim())) {
      return 'Parental Involvement must be Low, Medium or High';
    }
    if (!lowMedHigh.contains(_resourcesCtrl.text.trim())) {
      return 'Access to Resources must be Low, Medium or High';
    }
    if (!lowMedHigh.contains(_motivationCtrl.text.trim())) {
      return 'Motivation Level must be Low, Medium or High';
    }
    if (!lowMedHigh.contains(_incomeCtrl.text.trim())) {
      return 'Family Income must be Low, Medium or High';
    }
    if (!lowMedHigh.contains(_teacherCtrl.text.trim())) {
      return 'Teacher Quality must be Low, Medium or High';
    }
    if (!peerOpts.contains(_peerCtrl.text.trim())) {
      return 'Peer Influence must be Negative, Neutral or Positive';
    }
    if (!eduOpts.contains(_parentEduCtrl.text.trim())) {
      return 'Parental Education Level must be High School, College or Postgraduate';
    }
    if (!distOpts.contains(_distanceCtrl.text.trim())) {
      return 'Distance from Home must be Near, Moderate or Far';
    }
    if (!yesNo.contains(_extraCtrl.text.trim())) {
      return 'Extracurricular Activities must be Yes or No';
    }
    if (!yesNo.contains(_internetCtrl.text.trim())) {
      return 'Internet Access must be Yes or No';
    }
    if (!yesNo.contains(_disabilityCtrl.text.trim())) {
      return 'Learning Disabilities must be Yes or No';
    }

    return null;
  }

  Future<void> _predict() async {
    final missing = _checkMissing();
    if (missing != null) {
      setState(() {
        _isError = true;
        _resultText = missing;
      });
      return;
    }

    final rangeErr = _checkRanges();
    if (rangeErr != null) {
      setState(() {
        _isError = true;
        _resultText = rangeErr;
      });
      return;
    }

    setState(() {
      _loading = true;
      _resultText = '';
      _isError = false;
    });

    final body = {
      'Hours_Studied': double.parse(_hoursCtrl.text.trim()),
      'Attendance': double.parse(_attendanceCtrl.text.trim()),
      'Previous_Scores': double.parse(_prevScoresCtrl.text.trim()),
      'Tutoring_Sessions': int.parse(_tutoringCtrl.text.trim()),
      'Parental_Involvement': _parentalInvCtrl.text.trim(),
      'Access_to_Resources': _resourcesCtrl.text.trim(),
      'Motivation_Level': _motivationCtrl.text.trim(),
      'Family_Income': _incomeCtrl.text.trim(),
      'Teacher_Quality': _teacherCtrl.text.trim(),
      'Peer_Influence': _peerCtrl.text.trim(),
      'Parental_Education_Level': _parentEduCtrl.text.trim(),
      'Distance_from_Home': _distanceCtrl.text.trim(),
      'Extracurricular_Activities': _extraCtrl.text.trim(),
      'Internet_Access': _internetCtrl.text.trim(),
      'Learning_Disabilities': _disabilityCtrl.text.trim(),
    };

    try {
      final res = await http
          .post(
            Uri.parse('$apiBase/predict'),
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 120));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        final score = data['predicted_exam_score'];
        setState(() {
          _isError = false;
          _resultText = 'Predicted Exam Score: $score';
        });
      } else {
        String msg = 'Request failed (${res.statusCode})';
        try {
          final err = jsonDecode(res.body);
          if (err is Map && err['detail'] != null) {
            msg = err['detail'].toString();
          }
        } catch (_) {}
        setState(() {
          _isError = true;
          _resultText = msg;
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _resultText =
            'Could not reach the API. It may still be waking up — wait ~30s and try again.\n($e)';
      });
    } finally {
      setState(() => _loading = false);
    }
  }

  Widget _field(String label, TextEditingController ctrl,
      {String? hint, TextInputType? type}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextField(
        controller: ctrl,
        keyboardType: type,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Exam Score Predictor'),
        centerTitle: true,
        backgroundColor: Colors.teal.shade700,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Enter student details',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const SizedBox(height: 4),
              Text(
                'Fill every field, then tap Predict.',
                style: TextStyle(color: Colors.grey.shade700, fontSize: 13),
              ),
              const SizedBox(height: 16),
              _field('Hours Studied', _hoursCtrl,
                  hint: '0 - 60', type: TextInputType.number),
              _field('Attendance', _attendanceCtrl,
                  hint: '0 - 100', type: TextInputType.number),
              _field('Previous Scores', _prevScoresCtrl,
                  hint: '0 - 100', type: TextInputType.number),
              _field('Tutoring Sessions', _tutoringCtrl,
                  hint: '0 - 20', type: TextInputType.number),
              _field('Parental Involvement', _parentalInvCtrl,
                  hint: 'Low / Medium / High'),
              _field('Access to Resources', _resourcesCtrl,
                  hint: 'Low / Medium / High'),
              _field('Motivation Level', _motivationCtrl,
                  hint: 'Low / Medium / High'),
              _field('Family Income', _incomeCtrl,
                  hint: 'Low / Medium / High'),
              _field('Teacher Quality', _teacherCtrl,
                  hint: 'Low / Medium / High'),
              _field('Peer Influence', _peerCtrl,
                  hint: 'Negative / Neutral / Positive'),
              _field('Parental Education Level', _parentEduCtrl,
                  hint: 'High School / College / Postgraduate'),
              _field('Distance from Home', _distanceCtrl,
                  hint: 'Near / Moderate / Far'),
              _field('Extracurricular Activities', _extraCtrl,
                  hint: 'Yes / No'),
              _field('Internet Access', _internetCtrl, hint: 'Yes / No'),
              _field('Learning Disabilities', _disabilityCtrl,
                  hint: 'Yes / No'),
              const SizedBox(height: 8),
              SizedBox(
                height: 48,
                child: ElevatedButton(
                  onPressed: _loading ? null : _predict,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.teal.shade700,
                    foregroundColor: Colors.white,
                  ),
                  child: _loading
                      ? const SizedBox(
                          width: 22,
                          height: 22,
                          child: CircularProgressIndicator(
                            strokeWidth: 2.5,
                            color: Colors.white,
                          ),
                        )
                      : const Text(
                          'Predict',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                ),
              ),
              const SizedBox(height: 20),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: _resultText.isEmpty
                      ? Colors.grey.shade100
                      : (_isError
                          ? Colors.red.shade50
                          : Colors.teal.shade50),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: _resultText.isEmpty
                        ? Colors.grey.shade300
                        : (_isError
                            ? Colors.red.shade300
                            : Colors.teal.shade300),
                  ),
                ),
                child: Text(
                  _resultText.isEmpty
                      ? 'Prediction will show up here.'
                      : _resultText,
                  style: TextStyle(
                    fontSize: 15,
                    height: 1.35,
                    color: _resultText.isEmpty
                        ? Colors.grey.shade600
                        : (_isError
                            ? Colors.red.shade800
                            : Colors.teal.shade900),
                    fontWeight:
                        _resultText.isEmpty ? FontWeight.normal : FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
